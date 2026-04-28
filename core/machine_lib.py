import requests
from os import environ
from time import sleep
import time
import json
import pandas as pd
import random
import pickle
from itertools import product
from itertools import combinations
from collections import defaultdict
import pickle
import re

from .log_manager import setup_logger
logger = setup_logger(__name__, "miner")

arsenal = ["ts_moment", "ts_entropy", "ts_min_max_cps", "ts_min_max_diff", "inst_tvr", 'sigmoid', 
           "ts_decay_exp_window", "ts_percentage", "vector_neut", "vector_proj", "signed_power"]

group_ops = ["group_rank", "group_sum", "group_max", "group_mean", "group_median", "group_min", "group_std_dev"]

twin_field_ops = ["ts_corr", "ts_covariance", "ts_co_kurtosis", "ts_co_skewness", "ts_theilsen"]

class WorldQuantBrain:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = None
        self.basic_ops = ["log", "sqrt", "reverse", "inverse", "rank", "zscore", "log_diff", "s_log_1p",
                         'fraction', 'quantile', "normalize", "scale_down"]
        self.ts_ops = ["ts_rank", "ts_zscore", "ts_delta", "ts_sum", "ts_product",
                      "ts_ir", "ts_std_dev", "ts_mean", "ts_arg_min", "ts_arg_max", "ts_min_diff",
                      "ts_max_diff", "ts_returns", "ts_scale", "ts_skewness", "ts_kurtosis",  
                      "ts_quantile"]
        self.ops_set = self.basic_ops + self.ts_ops + arsenal + group_ops
        # Add list of known inaccessible operators
        self.inaccessible_ops = ["log_diff", "s_log_1p", "fraction", "quantile"]
        self.login()

    def login(self):
        from .api_session import create_session
        logger.info("Authenticating with WorldQuant Brain...")
        mgr = create_session(username=self.username, password=self.password)
        self.session = mgr.session
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        logger.info("Authentication successful")
        return self.session

    def _extract_inaccessible_operator(self, error_message: str) -> str:
        """Extract the inaccessible operator from error message."""
        match = re.search(r'operator "([^"]+)"', error_message)
        if match:
            return match.group(1)
        return None

    def _has_inaccessible_operator(self, alpha: str) -> bool:
        """Check if alpha expression contains any inaccessible operators."""
        for op in self.inaccessible_ops:
            if op in alpha:
                logger.warning(f"Skipping alpha with inaccessible operator '{op}': {alpha}")
                return True
        return False

    def single_simulate(self, alpha_data: list, neut: str, region: str, universe: str) -> dict:
        """Run a single alpha simulation."""
        logger.info(f"Starting single simulation for alpha")
        
        sim_data_list = self.generate_sim_data(alpha_data, region, universe, neut)
        results = []
        
        for sim_data in sim_data_list:
            try:
                # Skip simulation if alpha contains inaccessible operators
                if self._has_inaccessible_operator(sim_data['regular']):
                    continue
                    
                simulation_response = self.session.post('https://api.worldquantbrain.com/simulations', 
                                                     json=sim_data)
                if simulation_response.status_code == 401:
                    logger.info("Session expired, re-authenticating...")
                    self.login()
                    simulation_response = self.session.post('https://api.worldquantbrain.com/simulations', 
                                                        json=sim_data)
                
                if simulation_response.status_code != 201:
                    logger.error(f"Simulation API error: {simulation_response.text}")
                    continue
                    
                simulation_progress_url = simulation_response.headers.get('Location')
                if not simulation_progress_url:
                    logger.error("No Location header in response")
                    continue
                
                # Monitor this single simulation
                result = self._monitor_single_progress(simulation_progress_url)
                if result:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error in simulation: {str(e)}")
                sleep(60)  # Short sleep on error
                self.login()
                continue
                
        return results

    def _monitor_single_progress(self, progress_url: str) -> dict:
        """Monitor a single simulation's progress."""
        try:
            while True:
                simulation_progress = self.session.get(progress_url)
                retry_after = simulation_progress.headers.get("Retry-After", 0)
                
                if not retry_after:
                    result = simulation_progress.json()
                    status = result.get("status")
                    logger.info(f"Simulation status: {status}")
                    
                    if status == "COMPLETE":
                        return result
                    elif status in ["FAILED", "ERROR"]:
                        error_message = result.get("message", "")
                        logger.error(f"Simulation failed: {result}")
                        
                        # Check if error is due to inaccessible operator
                        inaccessible_op = self._extract_inaccessible_operator(error_message)
                        if inaccessible_op and inaccessible_op not in self.inaccessible_ops:
                            logger.info(f"Adding new inaccessible operator: {inaccessible_op}")
                            self.inaccessible_ops.append(inaccessible_op)
                        
                        return None
                    
                    break
                
                sleep(float(retry_after))
                
        except Exception as e:
            logger.error(f"Error monitoring progress: {str(e)}")
            return None

    def generate_sim_data(self, alpha_list, region, uni, neut):
        sim_data_list = []
        for alpha, decay in alpha_list:
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': uni,
                    'delay': 1,
                    'decay': decay,
                    'neutralization': neut,
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'OFF',
                    'language': 'FASTEXPR',
                    'visualization': False,
                },
                'regular': alpha}

            sim_data_list.append(simulation_data)
        return sim_data_list

    def locate_alpha(self, alpha_id):
        alpha = self.session.get("https://api.worldquantbrain.com/alphas/" + alpha_id)
        string = alpha.content.decode('utf-8')
        metrics = json.loads(string)
        #print(metrics["regular"]["code"])
        
        dateCreated = metrics["dateCreated"]
        sharpe = metrics["is"]["sharpe"]
        fitness = metrics["is"]["fitness"]
        turnover = metrics["is"]["turnover"]
        margin = metrics["is"]["margin"]
        
        triple = [sharpe, fitness, turnover, margin, dateCreated]
    
        return triple

    def set_alpha_properties(self,
        alpha_id,
        name: str = None,
        color: str = None,
        selection_desc: str = "None",
        combo_desc: str = "None",
        tags: str = ["ace_tag"],
    ):
        """
        Function changes alpha's description parameters
        """
    
        params = {
            "color": color,
            "name": name,
            "tags": tags,
            "category": None,
            "regular": {"description": None},
            "combo": {"description": combo_desc},
            "selection": {"description": selection_desc},
        }
        response = self.session.patch(
            "https://api.worldquantbrain.com/alphas/" + alpha_id, json=params
        )
    
    def check_submission(self, alpha_bag, gold_bag, start):
        depot = []
        for idx, g in enumerate(alpha_bag):
            if idx < start:
                continue
            if idx % 5 == 0:
                print(idx)
            if idx % 200 == 0:
                self.login()
            #print(idx)
            pc = self.get_check_submission(g)
            if pc == "sleep":
                sleep(100)
                self.login()
                alpha_bag.append(g)
            elif pc != pc:
                # pc is nan
                print("check self-corrlation error")
                sleep(100)
                alpha_bag.append(g)
            elif pc == "fail":
                continue
            elif pc == "error":
                depot.append(g)
            else:
                print(g)
                gold_bag.append((g, pc))
        print(depot)
        return gold_bag

    def get_check_submission(self, alpha_id):
        while True:
            result = self.session.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/check")
            if "retry-after" in result.headers:
                time.sleep(float(result.headers["Retry-After"]))
            else:
                break
        try:
            if result.json().get("is", 0) == 0:
                print("logged out")
                return "sleep"
            checks_df = pd.DataFrame(
                    result.json()["is"]["checks"]
            )
            pc = checks_df[checks_df.name == "PROD_CORRELATION"]["value"].values[0]
            if not any(checks_df["result"] == "FAIL"):
                return pc
            else:
                return "fail"
        except:
            print("catch: %s"%(alpha_id))
            return "error"
            
    def get_vec_fields(self, fields):

        vec_ops = ["vec_avg", "vec_sum", "vec_ir", "vec_max", "vec_count","vec_skewness","vec_stddev", "vec_choose"]
        vec_fields = []
     
        for field in fields:
            for vec_op in vec_ops:
                if vec_op == "vec_choose":
                    vec_fields.append("%s(%s, nth=-1)"%(vec_op, field))
                    vec_fields.append("%s(%s, nth=0)"%(vec_op, field))
                else:
                    vec_fields.append("%s(%s)"%(vec_op, field))
     
        return(vec_fields)

    def get_datafields(self,
        instrument_type: str = 'EQUITY',
        region: str = 'USA',
        delay: int = 1,
        universe: str = 'TOP3000',
        dataset_id: str = '',
        search: str = ''
    ):
        if len(search) == 0:
            url_template = "https://api.worldquantbrain.com/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
                "&offset={x}"
            count = self.session.get(url_template.format(x=0)).json()['count'] 
            
        else:
            url_template = "https://api.worldquantbrain.com/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
                f"&search={search}" +\
                "&offset={x}"
            count = 100
        
        datafields_list = []
        for x in range(0, count, 50):
            datafields = self.session.get(url_template.format(x=x))
            datafields_list.append(datafields.json()['results'])
     
        datafields_list_flat = [item for sublist in datafields_list for item in sublist]
     
        datafields_df = pd.DataFrame(datafields_list_flat)
        return datafields_df

    def process_datafields(self, df, data_type):

        if data_type == "matrix":
            datafields = df[df['type'] == "MATRIX"]["id"].tolist()
        elif data_type == "vector":
            datafields = self.get_vec_fields(df[df['type'] == "VECTOR"]["id"].tolist())

        tb_fields = []
        for field in datafields:
            tb_fields.append("winsorize(ts_backfill(%s, 120), std=4)"%field)
        return tb_fields
     
    def view_alphas(self, gold_bag):
        sharp_list = []
        for gold, pc in gold_bag:

            triple = self.locate_alpha(gold)
            info = [triple[2], triple[3], triple[4], triple[5], triple[6], triple[1]]
            info.append(pc)
            sharp_list.append(info)

        sharp_list.sort(reverse=True, key = lambda x : x[3])
        for i in sharp_list:
            print(i)
     
    def get_alphas(self, start_date, end_date, sharpe_th, fitness_th, region, alpha_num, usage):
        next_alphas = []
        decay_alphas = []
        # 3E large 3C less
        count = 0
        for i in range(0, alpha_num, 100):
            print(i)
            url_e = "https://api.worldquantbrain.com/users/self/alphas?limit=100&offset=%d"%(i) \
                    + "&status=UNSUBMITTED%1FIS_FAIL&dateCreated%3E=2025-" + start_date  \
                    + "T00:00:00-04:00&dateCreated%3C2025-" + end_date \
                    + "T00:00:00-04:00&is.fitness%3E" + str(fitness_th) + "&is.sharpe%3E" \
                    + str(sharpe_th) + "&settings.region=" + region + "&order=-is.sharpe&hidden=false&type!=SUPER"
            url_c = "https://api.worldquantbrain.com/users/self/alphas?limit=100&offset=%d"%(i) \
                    + "&status=UNSUBMITTED%1FIS_FAIL&dateCreated%3E=2025-" + start_date  \
                    + "T00:00:00-04:00&dateCreated%3C2025-" + end_date \
                    + "T00:00:00-04:00&is.fitness%3C-" + str(fitness_th) + "&is.sharpe%3C-" \
                    + str(sharpe_th) + "&settings.region=" + region + "&order=is.sharpe&hidden=false&type!=SUPER"
            urls = [url_e]
            if usage != "submit":
                urls.append(url_c)
            for url in urls:
                response = self.session.get(url)
                #print(response.json())
                try:
                    alpha_list = response.json()["results"]
                    #print(response.json())
                    for j in range(len(alpha_list)):
                        alpha_id = alpha_list[j]["id"]
                        name = alpha_list[j]["name"]
                        dateCreated = alpha_list[j]["dateCreated"]
                        sharpe = alpha_list[j]["is"]["sharpe"]
                        fitness = alpha_list[j]["is"]["fitness"]
                        turnover = alpha_list[j]["is"]["turnover"]
                        margin = alpha_list[j]["is"]["margin"]
                        longCount = alpha_list[j]["is"]["longCount"]
                        shortCount = alpha_list[j]["is"]["shortCount"]
                        decay = alpha_list[j]["settings"]["decay"]
                        exp = alpha_list[j]['regular']['code']
                        count += 1
                        #if (sharpe > 1.2 and sharpe < 1.6) or (sharpe < -1.2 and sharpe > -1.6):
                        if (longCount + shortCount) > 100:
                            if sharpe < -1.2:
                                exp = "-%s"%exp
                            rec = [alpha_id, exp, sharpe, turnover, fitness, margin, dateCreated, decay]
                            print(rec)
                            if turnover > 0.7:
                                rec.append(decay*4)
                                decay_alphas.append(rec)
                            elif turnover > 0.6:
                                rec.append(decay*3+3)
                                decay_alphas.append(rec)
                            elif turnover > 0.5:
                                rec.append(decay*3)
                                decay_alphas.append(rec)
                            elif turnover > 0.4:
                                rec.append(decay*2)
                                decay_alphas.append(rec)
                            elif turnover > 0.35:
                                rec.append(decay+4)
                                decay_alphas.append(rec)
                            elif turnover > 0.3:
                                rec.append(decay+2)
                                decay_alphas.append(rec)
                            else:
                                next_alphas.append(rec)
                except:
                    print("%d finished re-login"%i)
                    self.login()

        output_dict = {"next" : next_alphas, "decay" : decay_alphas}
        print("count: %d"%count)
        return output_dict
     
    def transform(self, next_alpha_recs, region):
        output = []
        for rec in next_alpha_recs:
            
            decay = rec[-1]
            exp = rec[1]
            output.append([exp,decay])
        output_dict = {region : output}
        return output_dict

    def prune(self, next_alpha_recs, region, prefix, keep_num):
        # prefix is the datafield prefix, fnd6, mdl175 ...
        # keep_num is the num of top sharpe same-datafield alpha
        output = []
        num_dict = defaultdict(int)
        for rec in next_alpha_recs:
            exp = rec[1]
            field = exp.split(prefix)[-1].split(",")[0]
            sharpe = rec[2]
            if sharpe < 0:
                field = "-%s"%field
            if num_dict[field] < keep_num:
                num_dict[field] += 1
                decay = rec[-1]
                exp = rec[1]
                output.append([exp,decay])
        output_dict = {region : output}
        return output_dict

    def get_first_order(self, vec_fields, ops_set):
        alpha_set = []
        for field in vec_fields:
            alpha_set.append(field)
            for op in ops_set:
                if op == "ts_percentage":
                    alpha_set += self.ts_comp_factory(op, field, "percentage", [0.5])
                elif op == "ts_decay_exp_window":
                    alpha_set += self.ts_comp_factory(op, field, "factor", [0.5])
                elif op == "ts_moment":
                    alpha_set += self.ts_comp_factory(op, field, "k", [2, 3, 4])
                elif op == "ts_entropy":
                    alpha_set += self.ts_comp_factory(op, field, "buckets", [10])
                elif op in twin_field_ops:
                    alpha_set += self.twin_field_factory(op, field, vec_fields)
                elif op.startswith("ts_") or op == "inst_tvr":
                    alpha_set += self.ts_factory(op, field)
                elif op.startswith("group_"):
                    alpha_set += self.group_factory(op, field, "usa")
                elif op.startswith("vector"):
                    alpha_set += self.vector_factory(op, field)
                elif op == "signed_power":
                    alpha = "%s(%s, 2)"%(op, field)
                    alpha_set.append(alpha)
                else:
                    alpha = "%s(%s)"%(op, field)
                    alpha_set.append(alpha)
        return alpha_set
        
    def get_group_second_order_factory(self, first_order, group_ops, region):
        second_order = []
        for fo in first_order:
            for group_op in group_ops:
                second_order += self.group_factory(group_op, fo, region)
        return second_order
     
    def get_ts_second_order_factory(self, first_order, ts_ops):
        second_order = []
        for fo in first_order:
            for ts_op in ts_ops:
                second_order += self.ts_factory(ts_op, fo)
        return second_order
     
     
    def get_data_fields_csv(self, filename, prefix):
        '''
        inputs: 
        CSV file with header 'field' 
        outputs:
        A list of string
        '''
        df = pd.read_csv(filename,header=0,encoding = 'unicode_escape')
        collection = []
        for _, row in df.iterrows():
            if row['field'].startswith(prefix):
                collection.append(row['field'])
     
        return collection
     
    def ts_arith_factory(self, ts_op, arith_op, field):
        first_order = "%s(%s)"%(arith_op, field)
        second_order = self.ts_factory(ts_op, first_order)
        return second_order
     
    def arith_ts_factory(self, arith_op, ts_op, field):
        second_order = []
        first_order = self.ts_factory(ts_op, field)
        for fo in first_order:
            second_order.append("%s(%s)"%(arith_op, fo))
        return second_order
     
    def ts_group_factory(self, ts_op, group_op, field, region):
        second_order = []
        first_order = self.group_factory(group_op, field, region)
        for fo in first_order:
            second_order += self.ts_factory(ts_op, fo)
        return second_order
     
    def group_ts_factory(self, group_op, ts_op, field, region):
        second_order = []
        first_order = self.ts_factory(ts_op, field)
        for fo in first_order:
            second_order += self.group_factory(group_op, fo, region)
        return second_order
     
    def vector_factory(self, op, field):
        output = []
        vectors = ["cap"]
        
        for vector in vectors:
        
            alpha = "%s(%s, %s)"%(op, field, vector)
            output.append(alpha)
        
        return output
     
    def trade_when_factory(self, op,field,region):
        output = []
        open_events = ["ts_arg_max(volume, 5) == 0", "ts_corr(close, volume, 20) < 0",
                       "ts_corr(close, volume, 5) < 0", "ts_mean(volume,10)>ts_mean(volume,60)",
                       "group_rank(ts_std_dev(returns,60), sector) > 0.7", "ts_zscore(returns,60) > 2",
                       "ts_skewness(returns,120)> 0.7", "ts_arg_min(volume, 5) > 3",
                       "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
                       "ts_arg_max(close, 5) == 0", "ts_arg_max(close, 20) == 0",
                       "ts_corr(close, volume, 5) > 0", "ts_corr(close, volume, 5) > 0.3", "ts_corr(close, volume, 5) > 0.5",
                       "ts_corr(close, volume, 20) > 0", "ts_corr(close, volume, 20) > 0.3", "ts_corr(close, volume, 20) > 0.5",
                       "ts_regression(returns, %s, 5, lag = 0, rettype = 2) > 0"%field,
                       "ts_regression(returns, %s, 20, lag = 0, rettype = 2) > 0"%field,
                       "ts_regression(returns, ts_step(20), 20, lag = 0, rettype = 2) > 0",
                       "ts_regression(returns, ts_step(5), 5, lag = 0, rettype = 2) > 0"]

        exit_events = ["abs(returns) > 0.1", "-1", "days_from_last_change(ern3_pre_reptime) > 20"]

        usa_events = ["rank(rp_css_business) > 0.8", "ts_rank(rp_css_business, 22) > 0.8", "rank(vec_avg(mws82_sentiment)) > 0.8",
                      "ts_rank(vec_avg(mws82_sentiment),22) > 0.8", "rank(vec_avg(nws48_ssc)) > 0.8",
                      "ts_rank(vec_avg(nws48_ssc),22) > 0.8", "rank(vec_avg(mws50_ssc)) > 0.8", "ts_rank(vec_avg(mws50_ssc),22) > 0.8",
                      "ts_rank(vec_sum(scl12_alltype_buzzvec),22) > 0.9", "pcr_oi_270 < 1", "pcr_oi_270 > 1",]

        asi_events = ["rank(vec_avg(mws38_score)) > 0.8", "ts_rank(vec_avg(mws38_score),22) > 0.8"]

        eur_events = ["rank(rp_css_business) > 0.8", "ts_rank(rp_css_business, 22) > 0.8",
                      "rank(vec_avg(oth429_research_reports_fundamental_keywords_4_method_2_pos)) > 0.8",
                      "ts_rank(vec_avg(oth429_research_reports_fundamental_keywords_4_method_2_pos),22) > 0.8",
                      "rank(vec_avg(mws84_sentiment)) > 0.8", "ts_rank(vec_avg(mws84_sentiment),22) > 0.8",
                      "rank(vec_avg(mws85_sentiment)) > 0.8", "ts_rank(vec_avg(mws85_sentiment),22) > 0.8",
                      "rank(mdl110_analyst_sentiment) > 0.8", "ts_rank(mdl110_analyst_sentiment, 22) > 0.8",
                      "rank(vec_avg(nws3_scores_posnormscr)) > 0.8",
                      "ts_rank(vec_avg(nws3_scores_posnormscr),22) > 0.8",
                      "rank(vec_avg(mws36_sentiment_words_positive)) > 0.8",
                      "ts_rank(vec_avg(mws36_sentiment_words_positive),22) > 0.8"]

        glb_events = ["rank(vec_avg(mdl109_news_sent_1m)) > 0.8",
                      "ts_rank(vec_avg(mdl109_news_sent_1m),22) > 0.8",
                      "rank(vec_avg(nws20_ssc)) > 0.8",
                      "ts_rank(vec_avg(nws20_ssc),22) > 0.8",
                      "vec_avg(nws20_ssc) > 0",
                      "rank(vec_avg(nws20_bee)) > 0.8",
                      "ts_rank(vec_avg(nws20_bee),22) > 0.8",
                      "rank(vec_avg(nws20_qmb)) > 0.8",
                      "ts_rank(vec_avg(nws20_qmb),22) > 0.8"]

        chn_events = ["rank(vec_avg(oth111_xueqiunaturaldaybasicdivisionstat_senti_conform)) > 0.8",
                      "ts_rank(vec_avg(oth111_xueqiunaturaldaybasicdivisionstat_senti_conform),22) > 0.8",
                      "rank(vec_avg(oth111_gubanaturaldaydevicedivisionstat_senti_conform)) > 0.8",
                      "ts_rank(vec_avg(oth111_gubanaturaldaydevicedivisionstat_senti_conform),22) > 0.8",
                      "rank(vec_avg(oth111_baragedivisionstat_regi_senti_conform)) > 0.8",
                      "ts_rank(vec_avg(oth111_baragedivisionstat_regi_senti_conform),22) > 0.8"]

        kor_events = ["rank(vec_avg(mdl110_analyst_sentiment)) > 0.8",
                      "ts_rank(vec_avg(mdl110_analyst_sentiment),22) > 0.8",
                      "rank(vec_avg(mws38_score)) > 0.8",
                      "ts_rank(vec_avg(mws38_score),22) > 0.8"]

        twn_events = ["rank(vec_avg(mdl109_news_sent_1m)) > 0.8",
                      "ts_rank(vec_avg(mdl109_news_sent_1m),22) > 0.8",
                      "rank(rp_ess_business) > 0.8",
                      "ts_rank(rp_ess_business,22) > 0.8"]

        for oe in open_events:
            for ee in exit_events:
                alpha = "%s(%s, %s, %s)"%(op, oe, field, ee)
                output.append(alpha)
        return output
     
    def ts_factory(self, op, field):
        output = []
        #days = [3, 5, 10, 20, 60, 120, 240]
        days = [5, 22, 66, 120, 240]
        
        for day in days:
        
            alpha = "%s(%s, %d)"%(op, field, day)
            output.append(alpha)
        
        return output
     
    def ts_comp_factory(self, op, field, factor, paras):
        output = []
        #l1, l2 = [3, 5, 10, 20, 60, 120, 240], paras
        l1, l2 = [5, 22, 66, 240], paras
        comb = list(product(l1, l2))
        
        for day,para in comb:
            
            if type(para) == float:
                alpha = "%s(%s, %d, %s=%.1f)"%(op, field, day, factor, para)
            elif type(para) == int:
                alpha = "%s(%s, %d, %s=%d)"%(op, field, day, factor, para)
            
            output.append(alpha)
        
        return output
     
    def twin_field_factory(self, op, field, fields):
        
        output = []
        #days = [3, 5, 10, 20, 60, 120, 240]
        days = [5, 22, 66, 240]
        outset = list(set(fields) - set([field]))
        
        for day in days:
            for counterpart in outset:
                alpha = "%s(%s, %s, %d)"%(op, field, counterpart, day)
                output.append(alpha)
        
        return output
     
     
    def group_factory(self, op, field, region):
        output = []
        vectors = ["cap"] 
        
        chn_group_13 = ['pv13_h_min2_sector', 'pv13_di_6l', 'pv13_rcsed_6l', 'pv13_di_5l', 'pv13_di_4l', 
                            'pv13_di_3l', 'pv13_di_2l', 'pv13_di_1l', 'pv13_parent', 'pv13_level']
        
        
        chn_group_1 = ['sta1_top3000c30','sta1_top3000c20','sta1_top3000c10','sta1_top3000c2','sta1_top3000c5']
        
        chn_group_2 = ['sta2_top3000_fact4_c10','sta2_top2000_fact4_c50','sta2_top3000_fact3_c20']
     
        chn_group_7 = ['oth171_region_sector_long_d1_sector', 'oth171_region_sector_short_d1_sector', 
                       'oth171_sector_long_d1_sector', 'oth171_sector_short_d1_sector']
        
        hkg_group_13 = ['pv13_10_f3_g2_minvol_1m_sector', 'pv13_10_minvol_1m_sector', 'pv13_20_minvol_1m_sector', 
                        'pv13_2_minvol_1m_sector', 'pv13_5_minvol_1m_sector', 'pv13_1l_scibr', 'pv13_3l_scibr',
                        'pv13_2l_scibr', 'pv13_4l_scibr', 'pv13_5l_scibr']
        
        hkg_group_1 = ['sta1_allc50','sta1_allc5','sta1_allxjp_513_c20','sta1_top2000xjp_513_c5']
        
        hkg_group_2 = ['sta2_all_xjp_513_all_fact4_c10','sta2_top2000_xjp_513_top2000_fact3_c10',
                       'sta2_allfactor_xjp_513_13','sta2_top2000_xjp_513_top2000_fact3_c20']
        
        hkg_group_8 = ['oth455_relation_n2v_p10_q50_w5_kmeans_cluster_5',
                         'oth455_relation_n2v_p10_q50_w4_kmeans_cluster_10',
                         'oth455_relation_n2v_p10_q50_w1_kmeans_cluster_20',
                         'oth455_partner_n2v_p50_q200_w4_kmeans_cluster_5', 
                         'oth455_partner_n2v_p10_q50_w4_pca_fact3_cluster_10',
                         'oth455_customer_n2v_p50_q50_w1_kmeans_cluster_5']
        
        twn_group_13 = ['pv13_2_minvol_1m_sector','pv13_20_minvol_1m_sector','pv13_10_minvol_1m_sector',
                        'pv13_5_minvol_1m_sector','pv13_10_f3_g2_minvol_1m_sector','pv13_5_f3_g2_minvol_1m_sector',
                        'pv13_2_f4_g3_minvol_1m_sector']
        
        twn_group_1 = ['sta1_allc50','sta1_allxjp_513_c50','sta1_allxjp_513_c20','sta1_allxjp_513_c2',
                       'sta1_allc20','sta1_allxjp_513_c5','sta1_allxjp_513_c10','sta1_allc2','sta1_allc5']
        
        twn_group_2 = ['sta2_allfactor_xjp_513_0','sta2_all_xjp_513_all_fact3_c20',
                       'sta2_all_xjp_513_all_fact4_c20','sta2_all_xjp_513_all_fact4_c50']
        
        twn_group_8 = ['oth455_relation_n2v_p50_q200_w1_pca_fact1_cluster_20',
                         'oth455_relation_n2v_p10_q50_w3_kmeans_cluster_20',
                         'oth455_relation_roam_w3_pca_fact2_cluster_5',
                         'oth455_relation_n2v_p50_q50_w2_pca_fact2_cluster_10', 
                         'oth455_relation_n2v_p10_q200_w5_pca_fact2_cluster_20',
                         'oth455_relation_n2v_p50_q50_w5_kmeans_cluster_5']
        
        usa_group_13 = ['pv13_h_min2_3000_sector','pv13_r2_min20_3000_sector','pv13_r2_min2_3000_sector',
                        'pv13_r2_min2_3000_sector', 'pv13_h_min2_focused_pureplay_3000_sector']
        
        usa_group_1 = ['sta1_top3000c50','sta1_allc20','sta1_allc10','sta1_top3000c20','sta1_allc5']
        
        usa_group_2 = ['sta2_top3000_fact3_c50','sta2_top3000_fact4_c20','sta2_top3000_fact4_c10']
        
        usa_group_3 = ['sta3_2_sector', 'sta3_3_sector', 'sta3_news_sector', 'sta3_peer_sector',
                       'sta3_pvgroup1_sector', 'sta3_pvgroup2_sector', 'sta3_pvgroup3_sector', 'sta3_sec_sector']
        
        usa_group_4 = ['rsk69_01c_1m', 'rsk69_57c_1m', 'rsk69_02c_2m', 'rsk69_5c_2m', 'rsk69_02c_1m',
                       'rsk69_05c_2m', 'rsk69_57c_2m', 'rsk69_5c_1m', 'rsk69_05c_1m', 'rsk69_01c_2m']
        
        usa_group_5 = ['anl52_2000_backfill_d1_05c', 'anl52_3000_d1_05c', 'anl52_3000_backfill_d1_02c', 
                       'anl52_3000_backfill_d1_5c', 'anl52_3000_backfill_d1_05c', 'anl52_3000_d1_5c']
        
        usa_group_6 = ['mdl10_group_name']
        
        usa_group_7 = ['oth171_region_sector_long_d1_sector', 'oth171_region_sector_short_d1_sector', 
                       'oth171_sector_long_d1_sector', 'oth171_sector_short_d1_sector']
        
        usa_group_8 = ['oth455_competitor_n2v_p10_q50_w1_kmeans_cluster_10',
                         'oth455_customer_n2v_p10_q50_w5_kmeans_cluster_10',
                         'oth455_relation_n2v_p50_q200_w5_kmeans_cluster_20',
                         'oth455_competitor_n2v_p50_q50_w3_kmeans_cluster_10', 
                         'oth455_relation_n2v_p50_q50_w3_pca_fact2_cluster_10', 
                         'oth455_partner_n2v_p10_q50_w2_pca_fact2_cluster_5',
                         'oth455_customer_n2v_p50_q50_w3_kmeans_cluster_5',
                         'oth455_competitor_n2v_p50_q200_w5_kmeans_cluster_20']
        
        
        asi_group_13 = ['pv13_20_minvol_1m_sector', 'pv13_5_f3_g2_minvol_1m_sector', 'pv13_10_f3_g2_minvol_1m_sector',
                        'pv13_2_f4_g3_minvol_1m_sector', 'pv13_10_minvol_1m_sector', 'pv13_5_minvol_1m_sector']
        
        asi_group_1 = ['sta1_allc50', 'sta1_allc10', 'sta1_minvol1mc50','sta1_minvol1mc20',
                       'sta1_minvol1m_normc20', 'sta1_minvol1m_normc50']
        
        asi_group_8 = ['oth455_partner_roam_w3_pca_fact1_cluster_5',
                       'oth455_relation_roam_w3_pca_fact1_cluster_20',
                       'oth455_relation_roam_w3_kmeans_cluster_20',
                       'oth455_relation_n2v_p10_q200_w5_pca_fact1_cluster_20',
                       'oth455_relation_n2v_p10_q200_w5_pca_fact1_cluster_20',
                       'oth455_competitor_n2v_p10_q200_w1_kmeans_cluster_10']
        
        jpn_group_1 = ['sta1_alljpn_513_c5', 'sta1_alljpn_513_c50', 'sta1_alljpn_513_c2', 'sta1_alljpn_513_c20']
        
        jpn_group_2 = ['sta2_top2000_jpn_513_top2000_fact3_c20', 'sta2_all_jpn_513_all_fact1_c5',
                       'sta2_allfactor_jpn_513_9', 'sta2_all_jpn_513_all_fact1_c10']
        
        jpn_group_8 = ['oth455_customer_n2v_p50_q50_w5_kmeans_cluster_10', 
                       'oth455_customer_n2v_p50_q50_w4_kmeans_cluster_10', 
                       'oth455_customer_n2v_p50_q50_w3_kmeans_cluster_10', 
                       'oth455_customer_n2v_p50_q50_w2_kmeans_cluster_10',
                       'oth455_customer_n2v_p50_q200_w5_kmeans_cluster_10',
                       'oth455_customer_n2v_p50_q200_w5_kmeans_cluster_10']
        
        jpn_group_13 = ['pv13_2_minvol_1m_sector', 'pv13_2_f4_g3_minvol_1m_sector', 'pv13_10_minvol_1m_sector',
                        'pv13_10_f3_g2_minvol_1m_sector', 'pv13_all_delay_1_parent', 'pv13_all_delay_1_level']
        
        kor_group_13 = ['pv13_10_f3_g2_minvol_1m_sector', 'pv13_5_minvol_1m_sector', 'pv13_5_f3_g2_minvol_1m_sector',
                        'pv13_2_minvol_1m_sector', 'pv13_20_minvol_1m_sector', 'pv13_2_f4_g3_minvol_1m_sector']
        
        kor_group_1 = ['sta1_allc20','sta1_allc50','sta1_allc2','sta1_allc10','sta1_minvol1mc50',
                       'sta1_allxjp_513_c10', 'sta1_top2000xjp_513_c50']
        
        kor_group_2 =['sta2_all_xjp_513_all_fact1_c50','sta2_top2000_xjp_513_top2000_fact2_c50',
                      'sta2_all_xjp_513_all_fact4_c50','sta2_all_xjp_513_all_fact4_c5']
        
        kor_group_8 = ['oth455_relation_n2v_p50_q200_w3_pca_fact3_cluster_5',
                         'oth455_relation_n2v_p50_q50_w4_pca_fact2_cluster_10',
                         'oth455_relation_n2v_p50_q200_w5_pca_fact2_cluster_5',
                         'oth455_relation_n2v_p50_q200_w4_kmeans_cluster_10', 
                         'oth455_relation_n2v_p10_q50_w1_kmeans_cluster_10', 
                         'oth455_relation_n2v_p50_q50_w5_pca_fact1_cluster_20']
        
        eur_group_13 = ['pv13_5_sector', 'pv13_2_sector', 'pv13_v3_3l_scibr', 'pv13_v3_2l_scibr', 'pv13_2l_scibr',
                        'pv13_52_sector', 'pv13_v3_6l_scibr', 'pv13_v3_4l_scibr', 'pv13_v3_1l_scibr']
        
        eur_group_1 = ['sta1_allc10', 'sta1_allc2', 'sta1_top1200c2', 'sta1_allc20', 'sta1_top1200c10']
        
        eur_group_2 = ['sta2_top1200_fact3_c50','sta2_top1200_fact3_c20','sta2_top1200_fact4_c50']
        
        eur_group_3 = ['sta3_6_sector', 'sta3_pvgroup4_sector', 'sta3_pvgroup5_sector']
        
        eur_group_7 = ['oth171_region_sector_long_d1_sector', 'oth171_region_sector_short_d1_sector', 
                       'oth171_sector_long_d1_sector', 'oth171_sector_short_d1_sector']
        
        eur_group_8 = ['oth455_relation_n2v_p50_q200_w3_pca_fact1_cluster_5',
                         'oth455_competitor_n2v_p50_q200_w4_kmeans_cluster_20',
                         'oth455_competitor_n2v_p50_q200_w5_pca_fact1_cluster_10', 
                         'oth455_competitor_roam_w4_pca_fact2_cluster_20', 
                         'oth455_relation_n2v_p10_q200_w2_pca_fact2_cluster_20', 
                         'oth455_competitor_roam_w2_pca_fact3_cluster_20']
        
        glb_group_13 = ["pv13_10_f2_g3_sector", "pv13_2_f3_g2_sector", "pv13_2_sector", "pv13_52_all_delay_1_sector"]
        
        glb_group_3 = ['sta3_2_sector', 'sta3_3_sector', 'sta3_news_sector', 'sta3_peer_sector',
                       'sta3_pvgroup1_sector', 'sta3_pvgroup2_sector', 'sta3_pvgroup3_sector', 'sta3_sec_sector']
        
        glb_group_1 = ['sta1_allc20', 'sta1_allc10', 'sta1_allc50', 'sta1_allc5']
        
        glb_group_2 = ['sta2_all_fact4_c50', 'sta2_all_fact4_c20', 'sta2_all_fact3_c20', 'sta2_all_fact4_c10']
        
        glb_group_13 = ['pv13_2_sector', 'pv13_10_sector', 'pv13_3l_scibr', 'pv13_2l_scibr', 'pv13_1l_scibr',
                        'pv13_52_minvol_1m_all_delay_1_sector','pv13_52_minvol_1m_sector','pv13_52_minvol_1m_sector']
        
        glb_group_7 = ['oth171_region_sector_long_d1_sector', 'oth171_region_sector_short_d1_sector', 
                       'oth171_sector_long_d1_sector', 'oth171_sector_short_d1_sector']  
        
        glb_group_8 = ['oth455_relation_n2v_p10_q200_w5_kmeans_cluster_5',
                         'oth455_relation_n2v_p10_q50_w2_kmeans_cluster_5',
                         'oth455_relation_n2v_p50_q200_w5_kmeans_cluster_5', 
                         'oth455_customer_n2v_p10_q50_w4_pca_fact3_cluster_20', 
                         'oth455_competitor_roam_w2_pca_fact1_cluster_10', 
                         'oth455_relation_n2v_p10_q200_w2_kmeans_cluster_5']
        
        amr_group_13 = ['pv13_4l_scibr', 'pv13_1l_scibr', 'pv13_hierarchy_min51_f1_sector',
                        'pv13_hierarchy_min2_600_sector', 'pv13_r2_min2_sector', 'pv13_h_min20_600_sector']
        
        amr_group_3 = ['sta3_news_sector', 'sta3_peer_sector', 'sta3_pvgroup1_sector', 'sta3_pvgroup2_sector',
                       'sta3_pvgroup3_sector']
        
        amr_group_8 = ['oth455_relation_roam_w1_pca_fact2_cluster_10', 
                       'oth455_competitor_n2v_p50_q50_w4_kmeans_cluster_10', 
                       'oth455_competitor_n2v_p50_q50_w3_kmeans_cluster_10', 
                       'oth455_competitor_n2v_p50_q50_w2_kmeans_cluster_10', 
                       'oth455_competitor_n2v_p50_q50_w1_kmeans_cluster_10',
                       'oth455_competitor_n2v_p50_q200_w5_kmeans_cluster_10']
        
        group_3 = ["oth171_region_sector_long_d1_sector", "oth171_region_sector_short_d1_sector",
                   "oth171_sector_long_d1_sector", "oth171_sector_short_d1_sector"]
        
        bps_group = "bucket(rank(fnd28_value_05480/close), range='0.2, 1, 0.2')"
        cap_group = "bucket(rank(cap), range='0.1, 1, 0.1')"
        sector_cap_group = "bucket(group_rank(cap,sector),range='0,1,0.1')"
        vol_group = "bucket(rank(ts_std_dev(ts_returns(close,1),20)),range = '0.1,1,0.1')"
        
        groups = ["market","sector", "industry", "subindustry", bps_group, cap_group, sector_cap_group]
        
        if region == "chn":
            groups += chn_group_13 + chn_group_1 + chn_group_2 + group_3 
        if region == "twn":
            groups += twn_group_13 + twn_group_1 + twn_group_2 + twn_group_8 
        if region == "asi":
            groups += asi_group_13 + asi_group_1 + asi_group_8 
        if region == "usa":
            groups += usa_group_13 + usa_group_1 + usa_group_2 + usa_group_3 + usa_group_4 + usa_group_8 + group_3 
            groups += usa_group_5 + usa_group_6 + usa_group_7
        if region == "hkg":
            groups += hkg_group_13 + hkg_group_1 + hkg_group_2 + hkg_group_8
        if region == "kor":
            groups += kor_group_13 + kor_group_1 + kor_group_2 + kor_group_8
        if region == "eur": 
            groups += eur_group_13 + eur_group_1 + eur_group_2 + eur_group_3 + eur_group_8 +  eur_group_7 + group_3 
        if region == "glb":
            groups += glb_group_13 + glb_group_8 + glb_group_3 + glb_group_1 + glb_group_7 + group_3
        if region == "amr":
            groups += amr_group_3 + amr_group_13
        if region == "jpn":
            groups += jpn_group_1 + jpn_group_2 + jpn_group_13 + jpn_group_8
            
        for group in groups:
            if op.startswith("group_vector"):
                for vector in vectors:
                    alpha = "%s(%s,%s,densify(%s))"%(op, field, vector, group)
                    output.append(alpha)
            elif op.startswith("group_percentage"):
                alpha = "%s(%s,densify(%s),percentage=0.5)"%(op, field, group)
                output.append(alpha)
            else:
                alpha = "%s(%s,densify(%s))"%(op, field, group)
                output.append(alpha)
        
        return output

    def load_task_pool(self, alpha_list: list, batch_size: int = 10, concurrent_batches: int = 10) -> list:
        """Split alpha list into pools of batches for concurrent processing."""
        pools = []
        current_pool = []
        current_batch = []
        
        for alpha in alpha_list:
            current_batch.append(alpha)
            
            if len(current_batch) >= batch_size:
                current_pool.append(current_batch)
                current_batch = []
                
                if len(current_pool) >= concurrent_batches:
                    pools.append(current_pool)
                    current_pool = []
        
        # Add any remaining batches
        if current_batch:
            current_pool.append(current_batch)
        if current_pool:
            pools.append(current_pool)
        
        logger.info(f"Created {len(pools)} pools with {batch_size} alphas per batch")
        return pools