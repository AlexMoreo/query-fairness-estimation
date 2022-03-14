Search.setIndex({docnames:["Datasets","Evaluation","Installation","Methods","Model-Selection","Plotting","index","modules","quapy","quapy.classification","quapy.data","quapy.method"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["Datasets.md","Evaluation.md","Installation.rst","Methods.md","Model-Selection.md","Plotting.md","index.rst","modules.rst","quapy.rst","quapy.classification.rst","quapy.data.rst","quapy.method.rst"],objects:{"":{quapy:[8,0,0,"-"]},"quapy.classification":{methods:[9,0,0,"-"],neural:[9,0,0,"-"],svmperf:[9,0,0,"-"]},"quapy.classification.methods":{LowRankLogisticRegression:[9,1,1,""]},"quapy.classification.methods.LowRankLogisticRegression":{fit:[9,2,1,""],get_params:[9,2,1,""],predict:[9,2,1,""],predict_proba:[9,2,1,""],set_params:[9,2,1,""],transform:[9,2,1,""]},"quapy.classification.neural":{CNNnet:[9,1,1,""],LSTMnet:[9,1,1,""],NeuralClassifierTrainer:[9,1,1,""],TextClassifierNet:[9,1,1,""],TorchDataset:[9,1,1,""]},"quapy.classification.neural.CNNnet":{document_embedding:[9,2,1,""],get_params:[9,2,1,""],vocabulary_size:[9,3,1,""]},"quapy.classification.neural.LSTMnet":{document_embedding:[9,2,1,""],get_params:[9,2,1,""],vocabulary_size:[9,3,1,""]},"quapy.classification.neural.NeuralClassifierTrainer":{device:[9,3,1,""],fit:[9,2,1,""],get_params:[9,2,1,""],predict:[9,2,1,""],predict_proba:[9,2,1,""],reset_net_params:[9,2,1,""],set_params:[9,2,1,""],transform:[9,2,1,""]},"quapy.classification.neural.TextClassifierNet":{dimensions:[9,2,1,""],document_embedding:[9,2,1,""],forward:[9,2,1,""],get_params:[9,2,1,""],predict_proba:[9,2,1,""],vocabulary_size:[9,3,1,""],xavier_uniform:[9,2,1,""]},"quapy.classification.neural.TorchDataset":{asDataloader:[9,2,1,""]},"quapy.classification.svmperf":{SVMperf:[9,1,1,""]},"quapy.classification.svmperf.SVMperf":{decision_function:[9,2,1,""],fit:[9,2,1,""],predict:[9,2,1,""],set_params:[9,2,1,""],valid_losses:[9,4,1,""]},"quapy.data":{base:[10,0,0,"-"],datasets:[10,0,0,"-"],preprocessing:[10,0,0,"-"],reader:[10,0,0,"-"]},"quapy.data.base":{Dataset:[10,1,1,""],LabelledCollection:[10,1,1,""],isbinary:[10,5,1,""]},"quapy.data.base.Dataset":{SplitStratified:[10,2,1,""],binary:[10,3,1,""],classes_:[10,3,1,""],kFCV:[10,2,1,""],load:[10,2,1,""],n_classes:[10,3,1,""],stats:[10,2,1,""],vocabulary_size:[10,3,1,""]},"quapy.data.base.LabelledCollection":{Xy:[10,3,1,""],artificial_sampling_generator:[10,2,1,""],artificial_sampling_index_generator:[10,2,1,""],binary:[10,3,1,""],counts:[10,2,1,""],kFCV:[10,2,1,""],load:[10,2,1,""],n_classes:[10,3,1,""],natural_sampling_generator:[10,2,1,""],natural_sampling_index_generator:[10,2,1,""],prevalence:[10,2,1,""],sampling:[10,2,1,""],sampling_from_index:[10,2,1,""],sampling_index:[10,2,1,""],split_stratified:[10,2,1,""],stats:[10,2,1,""],uniform_sampling:[10,2,1,""],uniform_sampling_index:[10,2,1,""]},"quapy.data.datasets":{fetch_UCIDataset:[10,5,1,""],fetch_UCILabelledCollection:[10,5,1,""],fetch_reviews:[10,5,1,""],fetch_twitter:[10,5,1,""],warn:[10,5,1,""]},"quapy.data.preprocessing":{IndexTransformer:[10,1,1,""],index:[10,5,1,""],reduce_columns:[10,5,1,""],standardize:[10,5,1,""],text2tfidf:[10,5,1,""]},"quapy.data.preprocessing.IndexTransformer":{add_word:[10,2,1,""],fit:[10,2,1,""],fit_transform:[10,2,1,""],transform:[10,2,1,""],vocabulary_size:[10,2,1,""]},"quapy.data.reader":{binarize:[10,5,1,""],from_csv:[10,5,1,""],from_sparse:[10,5,1,""],from_text:[10,5,1,""],reindex_labels:[10,5,1,""]},"quapy.error":{absolute_error:[8,5,1,""],acc_error:[8,5,1,""],acce:[8,5,1,""],ae:[8,5,1,""],f1_error:[8,5,1,""],f1e:[8,5,1,""],from_name:[8,5,1,""],kld:[8,5,1,""],mae:[8,5,1,""],mean_absolute_error:[8,5,1,""],mean_relative_absolute_error:[8,5,1,""],mkld:[8,5,1,""],mnkld:[8,5,1,""],mrae:[8,5,1,""],mse:[8,5,1,""],nkld:[8,5,1,""],rae:[8,5,1,""],relative_absolute_error:[8,5,1,""],se:[8,5,1,""],smooth:[8,5,1,""]},"quapy.evaluation":{artificial_prevalence_prediction:[8,5,1,""],artificial_prevalence_protocol:[8,5,1,""],artificial_prevalence_report:[8,5,1,""],evaluate:[8,5,1,""],gen_prevalence_prediction:[8,5,1,""],gen_prevalence_report:[8,5,1,""],natural_prevalence_prediction:[8,5,1,""],natural_prevalence_protocol:[8,5,1,""],natural_prevalence_report:[8,5,1,""]},"quapy.functional":{HellingerDistance:[8,5,1,""],adjusted_quantification:[8,5,1,""],artificial_prevalence_sampling:[8,5,1,""],get_nprevpoints_approximation:[8,5,1,""],normalize_prevalence:[8,5,1,""],num_prevalence_combinations:[8,5,1,""],prevalence_from_labels:[8,5,1,""],prevalence_from_probabilities:[8,5,1,""],prevalence_linspace:[8,5,1,""],strprev:[8,5,1,""],uniform_prevalence_sampling:[8,5,1,""],uniform_simplex_sampling:[8,5,1,""]},"quapy.method":{aggregative:[11,0,0,"-"],base:[11,0,0,"-"],meta:[11,0,0,"-"],neural:[11,0,0,"-"],non_aggregative:[11,0,0,"-"]},"quapy.method.aggregative":{ACC:[11,1,1,""],AdjustedClassifyAndCount:[11,4,1,""],AggregativeProbabilisticQuantifier:[11,1,1,""],AggregativeQuantifier:[11,1,1,""],CC:[11,1,1,""],ClassifyAndCount:[11,4,1,""],ELM:[11,1,1,""],EMQ:[11,1,1,""],ExpectationMaximizationQuantifier:[11,4,1,""],ExplicitLossMinimisation:[11,4,1,""],HDy:[11,1,1,""],HellingerDistanceY:[11,4,1,""],MAX:[11,1,1,""],MS2:[11,1,1,""],MS:[11,1,1,""],MedianSweep2:[11,4,1,""],MedianSweep:[11,4,1,""],OneVsAll:[11,1,1,""],PACC:[11,1,1,""],PCC:[11,1,1,""],ProbabilisticAdjustedClassifyAndCount:[11,4,1,""],ProbabilisticClassifyAndCount:[11,4,1,""],SLD:[11,4,1,""],SVMAE:[11,1,1,""],SVMKLD:[11,1,1,""],SVMNKLD:[11,1,1,""],SVMQ:[11,1,1,""],SVMRAE:[11,1,1,""],T50:[11,1,1,""],ThresholdOptimization:[11,1,1,""],X:[11,1,1,""]},"quapy.method.aggregative.ACC":{aggregate:[11,2,1,""],classify:[11,2,1,""],fit:[11,2,1,""],solve_adjustment:[11,2,1,""]},"quapy.method.aggregative.AggregativeProbabilisticQuantifier":{posterior_probabilities:[11,2,1,""],predict_proba:[11,2,1,""],probabilistic:[11,3,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.method.aggregative.AggregativeQuantifier":{aggregate:[11,2,1,""],aggregative:[11,3,1,""],classes_:[11,3,1,""],classify:[11,2,1,""],fit:[11,2,1,""],get_params:[11,2,1,""],learner:[11,3,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.method.aggregative.CC":{aggregate:[11,2,1,""],fit:[11,2,1,""]},"quapy.method.aggregative.ELM":{aggregate:[11,2,1,""],classify:[11,2,1,""],fit:[11,2,1,""]},"quapy.method.aggregative.EMQ":{EM:[11,2,1,""],EPSILON:[11,4,1,""],MAX_ITER:[11,4,1,""],aggregate:[11,2,1,""],fit:[11,2,1,""],predict_proba:[11,2,1,""]},"quapy.method.aggregative.HDy":{aggregate:[11,2,1,""],fit:[11,2,1,""]},"quapy.method.aggregative.OneVsAll":{aggregate:[11,2,1,""],binary:[11,3,1,""],classes_:[11,3,1,""],classify:[11,2,1,""],fit:[11,2,1,""],get_params:[11,2,1,""],posterior_probabilities:[11,2,1,""],probabilistic:[11,3,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.method.aggregative.PACC":{aggregate:[11,2,1,""],classify:[11,2,1,""],fit:[11,2,1,""]},"quapy.method.aggregative.PCC":{aggregate:[11,2,1,""],fit:[11,2,1,""]},"quapy.method.aggregative.ThresholdOptimization":{aggregate:[11,2,1,""],fit:[11,2,1,""]},"quapy.method.base":{BaseQuantifier:[11,1,1,""],BinaryQuantifier:[11,1,1,""],isaggregative:[11,5,1,""],isbinary:[11,5,1,""],isprobabilistic:[11,5,1,""]},"quapy.method.base.BaseQuantifier":{aggregative:[11,3,1,""],binary:[11,3,1,""],classes_:[11,3,1,""],fit:[11,2,1,""],get_params:[11,2,1,""],n_classes:[11,3,1,""],probabilistic:[11,3,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.method.base.BinaryQuantifier":{binary:[11,3,1,""]},"quapy.method.meta":{EACC:[11,5,1,""],ECC:[11,5,1,""],EEMQ:[11,5,1,""],EHDy:[11,5,1,""],EPACC:[11,5,1,""],Ensemble:[11,1,1,""],ensembleFactory:[11,5,1,""],get_probability_distribution:[11,5,1,""]},"quapy.method.meta.Ensemble":{VALID_POLICIES:[11,4,1,""],aggregative:[11,3,1,""],binary:[11,3,1,""],classes_:[11,3,1,""],fit:[11,2,1,""],get_params:[11,2,1,""],probabilistic:[11,3,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.method.neural":{QuaNetModule:[11,1,1,""],QuaNetTrainer:[11,1,1,""],mae_loss:[11,5,1,""]},"quapy.method.neural.QuaNetModule":{device:[11,3,1,""],forward:[11,2,1,""]},"quapy.method.neural.QuaNetTrainer":{classes_:[11,3,1,""],clean_checkpoint:[11,2,1,""],clean_checkpoint_dir:[11,2,1,""],fit:[11,2,1,""],get_params:[11,2,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.method.non_aggregative":{MaximumLikelihoodPrevalenceEstimation:[11,1,1,""]},"quapy.method.non_aggregative.MaximumLikelihoodPrevalenceEstimation":{classes_:[11,3,1,""],fit:[11,2,1,""],get_params:[11,2,1,""],quantify:[11,2,1,""],set_params:[11,2,1,""]},"quapy.model_selection":{GridSearchQ:[8,1,1,""]},"quapy.model_selection.GridSearchQ":{best_model:[8,2,1,""],classes_:[8,3,1,""],fit:[8,2,1,""],get_params:[8,2,1,""],quantify:[8,2,1,""],set_params:[8,2,1,""]},"quapy.plot":{binary_bias_bins:[8,5,1,""],binary_bias_global:[8,5,1,""],binary_diagonal:[8,5,1,""],brokenbar_supremacy_by_drift:[8,5,1,""],error_by_drift:[8,5,1,""]},"quapy.util":{EarlyStop:[8,1,1,""],create_if_not_exist:[8,5,1,""],create_parent_dir:[8,5,1,""],download_file:[8,5,1,""],download_file_if_not_exists:[8,5,1,""],get_quapy_home:[8,5,1,""],map_parallel:[8,5,1,""],parallel:[8,5,1,""],pickled_resource:[8,5,1,""],save_text_file:[8,5,1,""],temp_seed:[8,5,1,""]},quapy:{classification:[9,0,0,"-"],data:[10,0,0,"-"],error:[8,0,0,"-"],evaluation:[8,0,0,"-"],functional:[8,0,0,"-"],isbinary:[8,5,1,""],method:[11,0,0,"-"],model_selection:[8,0,0,"-"],plot:[8,0,0,"-"],util:[8,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:attribute","5":"py:function"},terms:{"0":[0,1,3,4,5,8,9,10,11],"00":[0,1,4,8],"000":1,"0001":[4,11],"000e":1,"001":[4,9,11],"005":8,"008":[],"009":1,"0097":[],"01":[8,9,11],"017":1,"018":0,"02":1,"021":0,"02552":4,"03":1,"034":1,"035":1,"037":1,"04":1,"041":1,"042":1,"046":1,"048":1,"05":[5,8,10],"055":1,"063":[0,10],"065":0,"070":1,"073":1,"075":1,"078":0,"081":[0,10],"082":[0,1],"083":0,"086":0,"091":1,"099":0,"1":[0,1,3,4,5,8,9,10,11],"10":[0,1,4,5,8,9,11],"100":[0,1,3,4,5,9,10,11],"1000":[0,4,11],"10000":4,"100000":4,"1007":[],"101":[4,8,10],"1010":4,"1024":11,"104":0,"108":1,"109":0,"11":[0,1,6,8,10],"11338":0,"114":1,"1145":[],"12":9,"120":0,"1215742":0,"1271":0,"13":[0,9],"139":0,"14":3,"142":1,"146":3,"1473":0,"148":0,"1484":0,"15":[3,8,10],"150":0,"153":0,"157":0,"158":0,"159":0,"1593":0,"1594":0,"1599":0,"161":0,"163":[0,1],"164":[0,3],"167":0,"17":0,"1771":1,"1775":[0,3],"1778":[0,3],"178":0,"1823":0,"1839":0,"18399":0,"1853":0,"19":[3,10],"193":0,"199151":0,"19982":4,"1e":9,"1st":0,"2":[0,1,3,5,8,10,11],"20":[5,8,11],"200":[1,9],"2000":0,"2002":3,"2006":11,"2008":11,"2011":4,"2013":3,"2015":[0,2,3,9,11],"2016":[3,10,11],"2017":[0,3,10,11],"2018":[0,3,10],"2019":[3,10,11],"2020":4,"2021":11,"20342":4,"206":0,"207":0,"208":0,"21":[1,3,5,8,10],"210":[],"211":0,"2126":0,"2155":0,"21591":[0,10],"218":3,"2184":0,"219e":1,"22":[0,3,9,10],"222":0,"222046":0,"226":0,"229":1,"229399":0,"23":9,"235":1,"238":0,"2390":0,"24":[0,9],"243":0,"248563":0,"24866":4,"24987":4,"25":[0,5,8,9,11],"25000":0,"256":[0,9],"26":9,"261":0,"265":0,"266":0,"267":0,"27":[1,3,9],"270":0,"2700406":[],"271":0,"272":0,"274":0,"275":1,"27th":[0,3,10],"28":3,"280":0,"281":0,"282":0,"283":[0,1],"288":0,"289":0,"2971":0,"2nd":0,"2t":[1,8],"2tp":8,"2x5fcv":0,"3":[0,1,3,5,6,8,9,10,11],"30":[0,1,3,11],"300":[0,1,9],"305":0,"306":0,"312":0,"32":[0,6],"3227":8,"3269206":[],"3269287":[],"33":[0,5,8],"331":0,"333":0,"335":0,"337":0,"34":[0,3,10,11],"341":0,"346":1,"347":0,"350":0,"351":0,"357":1,"359":0,"361":0,"366":1,"372":0,"373":0,"376132":0,"3765":0,"3813":0,"3821":[0,10],"383e":1,"387e":1,"392":0,"394":0,"399":0,"3f":[1,6],"3rd":0,"4":[0,1,3,4,5,8,11],"40":[0,3,4,11],"404333":0,"407":0,"41":3,"412":0,"412e":1,"413":0,"414":0,"417":0,"41734":4,"42":[1,8],"421":0,"4259":0,"426e":1,"427":0,"430":0,"434":0,"435":1,"43676":4,"437":0,"44":0,"4403":10,"446":0,"45":[3,5,10],"452":0,"459":1,"4601":0,"461":0,"463":0,"465":0,"466":0,"470":0,"48":3,"481":0,"48135":4,"486":0,"4898":0,"492":0,"496":0,"4960":1,"497":0,"5":[0,1,3,4,5,8,9,10,11],"50":[0,5,8,11],"500":[0,1,4,5,11],"5000":[1,5],"5005":4,"507":0,"508":0,"512":[9,11],"514":0,"515e":1,"530":0,"534":0,"535":0,"535e":1,"5379":4,"539":0,"541":1,"546":0,"5473":0,"54it":4,"55":5,"55it":4,"565":1,"569":0,"57":0,"573":0,"578":1,"583":0,"591":3,"5f":4,"5fcv":[],"5fcvx2":10,"6":[0,1,3,5,8,10],"60":0,"600":1,"601":0,"604":3,"606":0,"625":0,"627":0,"633e":1,"634":1,"64":[9,11],"640":0,"641":0,"650":0,"653":0,"654":1,"66":[1,11],"665":0,"667":0,"669":0,"67":[5,8],"683":0,"688":0,"691":0,"694582":0,"7":[1,5,8,9,11],"70":0,"700":0,"701e":1,"711":0,"717":1,"725":1,"730":0,"735":0,"740e":1,"748":0,"75":[0,5,8],"762":0,"774":0,"778":0,"787":0,"794":0,"798":0,"8":[0,1,5,10,11],"8000":0,"830":0,"837":1,"858":1,"861":0,"87":[0,3,10],"8788":0,"889504":0,"8d2fhsgcvn0aaaaa":[],"9":[0,1,3,5,8],"90":[5,8],"901":0,"909":1,"914":1,"917":0,"919":[0,10],"922":0,"923":0,"935":0,"936":0,"937":[0,10],"945":1,"95":[8,10],"9533":0,"958":0,"97":0,"979":0,"982":0,"99":8,"abstract":[3,9,10,11],"boolean":[8,10,11],"case":[0,1,3,4,5,8,10,11],"class":[0,1,3,4,5,6,8,9,10,11],"d\u00edez":3,"default":[1,3,8,9,10,11],"do":[0,1,3,4,8,9,10,11],"final":[1,3,5,11],"float":[0,3,8,9,10,11],"function":[0,1,3,4,5,6,7,9,10,11],"g\u00e1llego":[0,3,10,11],"gonz\u00e1lez":3,"import":[0,1,3,4,5,6,10,11],"int":[0,5,8,10,11],"long":[4,9],"new":[0,3,10],"p\u00e9rez":[0,3,10,11],"return":[0,1,3,4,5,8,9,10,11],"rodr\u0131":3,"short":9,"static":[3,11],"true":[0,1,3,4,5,6,8,9,10,11],"try":4,"while":[3,5,8,9,10,11],A:[0,3,8,9,10,11],As:[3,4],By:[1,3,8],For:[0,1,5,6,8,10],If:[3,5,8,10,11],In:[0,1,2,3,4,5,6,9],It:[3,4,5,8],One:[0,1,3,11],That:[1,4],The:[0,1,2,4,5,6,8,9,10,11],Then:3,These:0,To:[5,10],_:[5,8,10],__:[],__class__:5,__name__:5,_adjust:[],_ae_:[],_classify_:[],_error_name_:[],_fit_learner_:[],_kld_:[],_labelledcollection_:[],_learner_:[],_mean:[],_min_df_:[],_my:[],_nkld_:[],_posterior_probabilities_:11,_q_:[],_rae_:[],_svmperf_:[],ab:[],aboud:3,about:[0,5,8,10],abov:[0,3,5,8],absolut:[1,3,5,6,8,11],absolute_error:8,abstractmethod:3,acc:[1,3,5,6,8,11],acc_error:8,accept:3,access:[0,3,10,11],accommod:0,accord:[1,3,4,8,9,10,11],accordingli:5,accuraci:[1,5,8,11],accuracy_polici:[],achiev:[1,3,4,5],acm:[0,3,10],across:[0,1,4,5,6,8],action:0,actual:[10,11],acut:0,ad:6,adapt:8,add:[3,4,8,10],add_word:10,addit:3,addition:0,adjust:[3,6,8,11],adjusted_quantif:8,adjustedclassifyandcount:11,adopt:[3,4,10],advanc:[0,6],advantag:[3,11],ae:[1,2,5,8,11],ae_:1,affect:8,after:[8,11],afterward:11,again:5,against:5,aggreg:[1,4,5,6,7,8],aggregativeprobabilisticquantifi:[3,11],aggregativequantifi:[3,11],aggregg:[],aim:[4,5],aka:[10,11],al:[0,2,9,10,11],alaiz:3,alegr:3,alejandro:4,algorithm:[8,11],alia:[3,8,11],all:[0,1,2,3,5,8,10,11],allia:3,alloc:[8,9],allow:[0,1,2,3,5,8,9,10,11],almost:3,along:[0,3,8,11],alreadi:[3,11],also:[0,1,2,3,5,6,8,9],altern:4,although:[3,4,5,11],alwai:[3,4,5,11],among:3,amount:8,an:[0,1,2,3,4,5,6,8,9,10,11],analys:[5,6],analysi:[0,3,6,10],analyz:5,ani:[0,1,3,4,5,6,8,9,10,11],anoth:[0,1,3,5],anotherdir:8,anyon:0,anyth:11,api:6,app:[8,10,11],appeal:1,appear:5,append:5,appli:[2,3,4,5,8,9,10,11],appropri:4,approxim:[1,5,8,9,10],ar:[0,1,3,4,5,8,9,10,11],archive_filenam:8,archive_path:[],arg:[8,10],argmax:8,args_i:8,argu:4,argument:[0,1,3,5,8,10,11],arifici:[],aris:1,around:[1,10],arrai:[1,3,5,8,9,10,11],articl:[3,4],artifici:[0,1,3,4,5,6,8,10],artificial_prevalence_predict:8,artificial_prevalence_protocol:8,artificial_prevalence_report:8,artificial_prevalence_sampl:8,artificial_sampling_ev:[1,4],artificial_sampling_gener:[0,10],artificial_sampling_index_gener:10,artificial_sampling_predict:[1,5],artificial_sampling_report:1,arxiv:4,asarrai:1,asdataload:9,asonam:0,assert:10,assess:4,assign:[3,8,10],associ:[8,10],assum:[1,6,11],assumpion:11,assumpt:[1,5,6],astyp:[],attempt:[3,11],attribut:11,august:0,autom:[0,3,6],automat:[0,1],av:[3,11],avail:[0,1,2,3,5,6,9,11],averag:[1,3,8,10,11],avoid:[1,8],ax:11,axi:[5,8],b:[0,10,11],balanc:[0,4,11],band:[5,8],bar:8,barranquero:[2,3,9,11],base:[0,3,6,7,8,9],base_classifi:5,base_estim:3,base_quantifier_class:11,baseestim:[9,11],baselin:6,basequantifi:[3,8,11],basic:[5,11],batch:9,batch_siz:9,batch_size_test:9,beat:11,been:[0,3,4,5,8,10,11],befor:[3,8,9,10,11],beforehand:8,behav:[3,5],being:[4,8,11],belief:1,belong:[3,11],below:[0,2,3,5,8,10],best:[4,8,9],best_epoch:8,best_model:8,best_model_:4,best_params_:4,best_scor:8,better:4,between:[4,5,6,8,9,11],beyond:5,bia:[6,8],bias:5,bidirect:11,bin:[5,8,11],bin_bia:5,bin_diag:5,binar:[8,10],binari:[3,5,6,8,9,10,11],binary_bias_bin:[5,8],binary_bias_glob:[5,8],binary_diagon:[5,8],binary_quantifi:11,binaryquantifi:11,binom:8,block:[0,8],bool:8,both:5,bound:[8,11],box:[5,8],breast:0,brief:1,bring:11,broken:[5,8],brokenbar_supremacy_by_drift:8,budg:1,budget:[1,4],build:[],bypass:11,c:[3,4,8,9,10,11],calcul:8,calibr:3,calibratedclassifi:3,calibratedclassifiercv:3,calibratedcv:[],call:[0,1,5,8,10,11],callabl:[0,8,10],can:[0,1,2,3,4,5,8,10,11],cancer:0,cannot:[],cardiotocographi:0,care:11,carri:[3,10,11],casa_token:[],castano:[3,10],castro:3,categor:[3,10],categori:[1,8],cc:[3,5,11],ceil:8,cell:11,center:5,chang:[0,1,3,10],character:[3,6],characteriz:[0,3,10],charg:[0,8,10],chart:8,check:[3,4],checkpoint:[9,11],checkpointdir:11,checkpointnam:11,checkpointpath:9,choic:4,choos:11,chosen:[4,8],cl:0,cla:[],class2int:10,class_weight:[4,11],classes_:[8,10,11],classif:[0,1,3,7,8,10,11],classif_posterior:[3,11],classif_predict:[3,11],classif_predictions_bin:11,classifi:[1,4,5,6,8,9,11],classifier_net:9,classifiermixin:9,classifyandcount:[3,11],classmethod:[0,10,11],classnam:10,classs:8,clean_checkpoint:11,clean_checkpoint_dir:11,clear:5,clearer:1,clearli:5,clip:8,close:[1,10],closer:1,closest:11,cm:8,cmc:0,cnn:[3,11],cnnnet:[3,9,11],code:[0,3,4,5,9],codifi:10,coincid:[0,6],col:[0,10],collect:[0,8,9,10],collet:10,color:[5,8],colormap:8,column:[0,8,10],com:8,combin:[0,1,4,8,10,11],combinatio:8,combinations_budget:8,come:[0,8,10,11],commandlin:[],common:11,commonli:6,compar:[5,8],comparison:5,compat:11,compil:[2,3],complement:11,complet:[3,5,11],compon:[8,9],compress:0,comput:[1,3,5,8,11],computation:4,compute_fpr:[],compute_t:[],compute_tpr:[],concept:6,concur:[],condit:[8,11],conduct:[0,8],confer:[0,3,10],confid:8,configur:[4,8],conform:10,connect:11,consecut:[8,9,11],consid:[3,5,8,9,10,11],consist:[0,4,5,8,9,10,11],constrain:[1,5,8,10],constructor:3,consult:[0,1],contain:[1,2,3,5,8,9,10,11],contanin:8,content:7,context:8,contrast:1,control:[1,4,10],conv_block:[],conv_lay:[],conveni:8,converg:11,convert:[1,3,8,9,10,11],convolut:9,copi:[8,10],cornel:[],correct:11,correctli:8,correspond:[5,8,10],cosest:11,cost:1,costli:4,could:[0,1,3,4,5,6],count:[4,5,6,8,10,11],count_:[],counter:10,countvector:10,covari:10,cover:[1,4,9],coz:[0,3,10],cpu:[1,9,11],creat:[0,6,8],create_if_not_exist:8,create_parent_dir:8,crisp:[3,8],criteria:4,cross:[3,10,11],cs:8,csr:10,csr_matrix:10,csv:10,ctg:0,cuda:[3,9,11],cumbersom:1,cumberson:8,cumul:11,curios:5,current:[3,8,9,10,11],custom:[3,6,8,10],customarili:[3,4],cv:[3,4],cyan:5,d:11,d_:8,dat:[0,9],data:[1,3,4,5,6,7,8,9,11],data_hom:10,datafram:[1,8],dataload:9,dataset:[1,3,4,5,6,7,8,9,11],dataset_nam:10,deal:0,decaesteck:[3,11],decai:9,decid:10,decim:1,decis:[3,8,9,11],decision_funct:9,decomposit:9,dedic:[1,10],deep:[3,8,11],def:[0,1,3,5,8],defin:[0,3,8,9,10,11],degre:4,del:[0,3,10],delai:8,deliv:[3,11],denomin:11,dens:[0,11],densiti:8,depend:[0,1,4,5,8,11],describ:[3,8,11],descript:0,design:4,desir:[0,1,10],despit:1,destin:8,detail:[0,1,3,6,9,10,11],determin:[1,4,5],detriment:5,devel:10,develop:[4,6],deviat:[0,1,5,8,10],devic:[0,3,5,9,11],df:1,df_replac:[],diabet:0,diagon:[6,8],dict:[8,10,11],dictionari:[8,9,10,11],differ:[0,1,3,4,5,6,8,10,11],difficult:5,digit:0,dimens:[8,9,10,11],dimension:[8,9,10,11],dir:8,directli:[0,1,3],directori:[2,8,9,10,11],discard:8,discoveri:3,discret:8,discuss:5,disjoint:9,disk:8,displai:[1,5,8],displaystyl:8,distanc:[8,11],distant:[1,8],distribut:[0,3,5,8,10,11],diverg:[1,3,8,11],divid:8,dl:[],doabl:0,doc_embed:11,doc_embedding_s:11,doc_posterior:11,document:[0,1,3,5,9,10,11],document_embed:9,doe:[0,2,3,8,11],doi:[],done:3,dot:[5,8],dowload:8,down:[5,8,10],download:[0,2,3,8],download_fil:8,download_file_if_not_exist:8,draw:[8,10],drawn:[0,1,4,8,10],drift:6,drop:9,drop_p:9,dropout:[9,11],ds:[3,11],ds_polici:[],ds_policy_get_posterior:[],dtype:[1,10],dump:10,dure:[1,5,11],dynam:[3,9,10,11],e:[0,1,3,4,5,6,8,9,10,11],eacc:11,each:[0,1,3,4,5,8,9,10,11],earli:[8,9,11],early_stop:[],earlystop:8,easili:[0,2,5,9],ecc:11,edu:[],eemq:11,effect:3,effici:3,ehdi:11,either:[1,3,8,10,11],element:[3,10,11],elm:[3,11],els:11,em:11,emb:9,embed:[3,9,11],embed_s:9,embedding_s:9,empti:10,emq:[5,11],enabl:9,encod:10,end:[4,8,11],endeavour:6,enough:5,ensembl:[0,6,10,11],ensemblefactori:11,ensure_probabilist:[],entir:[0,3,4,5,8],entri:11,environ:[1,3,4,5,8,11],ep:[1,8],epacc:11,epoch:[8,9,11],epsilon:[1,8,11],equal:[1,8],equidist:[0,8],equip:[3,5],equival:11,err:[],err_drift:5,err_nam:8,error:[3,4,6,7,9,11],error_:[],error_by_drift:[5,8],error_funct:1,error_metr:[1,4,8],error_nam:[5,8],especi:8,establish:8,estim:[1,3,5,6,8,9,10,11],estim_prev:[1,5,8],estim_preval:[3,6,11],estimant:11,esuli:[0,2,3,9,10,11],et:[0,2,9,10,11],etc:6,eval_budget:[4,8],evalu:[0,3,4,5,6,7,9,10,11],even:8,eventu:[9,10],everi:[3,11],everyth:3,evinc:5,ex:[],exact:[0,10],exactli:0,exampl:[0,1,3,4,5,8,9,10,11],exce:8,excel:0,except:[3,8,11],exemplifi:0,exhaust:8,exhibit:[4,5],exist:8,exist_ok:8,expand_frame_repr:1,expect:[6,11],expectationmaximizationquantifi:[3,11],experi:[1,2,3,4,5,8],explain:[1,5],explicit:11,explicitlossminim:[],explicitlossminimis:11,explor:[4,8,10],express:10,ext:2,extend:[2,3,11],extens:[0,2,5],extern:3,extract:[1,8,10],f1:[1,8,9],f1_error:8,f1e:[1,8],f:[0,1,3,4,5,6,10],f_1:8,fabrizio:4,facilit:6,fact:[3,5],factor:8,factori:11,fals:[1,3,5,8,9,10,11],famili:[3,11],familiar:3,far:[8,9,10],fare:8,fast:8,faster:[0,10],feat1:10,feat2:10,featn:10,featur:[0,10],feature_extract:10,fetch:[0,6],fetch_review:[0,1,3,4,5,10,11],fetch_twitt:[0,3,6,10],fetch_ucidataset:[0,3,10],fetch_ucilabelledcollect:[0,10],ff:11,ff_layer:11,fhe:0,file:[0,5,8,9,10,11],filenam:8,fin:0,find:[0,4],finish:4,first:[0,1,2,3,5,8,10,11],fit:[1,3,4,5,6,8,9,10,11],fit_learn:[3,11],fit_transform:10,fix:[1,4],flag:8,float64:1,fn:8,fold:[3,10,11],folder:[0,11],follow:[0,1,3,4,5,6,8,11],fomart:10,for_model_select:[0,10],form:[0,8,10],forman:11,format:[0,5,10],former:[2,11],forward:[9,11],found:[0,3,4,8,9,10],four:3,fp:8,fpr:[8,11],frac:8,framework:6,frequenc:[0,10,11],from:[0,1,3,4,5,6,8,10,11],from_csv:10,from_nam:[1,8],from_spars:10,from_text:10,full:[1,8],fulli:0,func:8,further:[0,1,3,9,10,11],fusion:[0,3,10],futur:3,g:[0,1,3,4,6,8,10,11],gain:8,gao:[0,3,10,11],gap:10,gasp:[0,10],gen:8,gen_data:5,gen_fn:8,gen_prevalence_predict:8,gen_prevalence_report:8,gener:[0,1,3,4,5,8,9,10,11],generation_func:8,german:0,get:[0,1,5,8,9,10,11],get_aggregative_estim:[],get_nprevpoints_approxim:[1,8],get_param:[3,8,9,11],get_probability_distribut:11,get_quapy_hom:8,ggener:8,github:[],give:11,given:[1,3,4,8,9,10,11],global:8,goal:11,goe:4,good:[4,5],got:4,govern:1,gpu:[9,11],grant:[],greater:10,grid:[4,8,10,11],gridsearchcv:[4,11],gridsearchq:[4,8,11],ground:11,group:3,guarante:10,guez:3,gzip:0,ha:[3,4,5,8,9,10,11],haberman:[0,3],had:10,handl:0,happen:[4,5],hard:3,harder:5,harmon:8,harri:0,hat:8,have:[0,1,2,3,4,5,8,10,11],hcr:[0,3,10],hd:8,hdy:[6,11],held:[3,4,8,9,11],helling:11,hellingerdist:8,hellingerdistancei:[3,11],hellingh:8,help:5,henc:[8,10],here:[1,11],heurist:11,hidden:[5,9,11],hidden_s:9,hide:5,high:[5,8],higher:[1,5],highlight:8,hightlight:8,histogram:11,hlt:[],hold:[6,8,11],home:[8,10],hook:11,how:[0,1,3,4,5,8,10,11],howev:[0,4,5],hp:[0,3,4,10],html:10,http:[8,10],hyper:[4,8,9],hyperparam:4,hyperparamet:[3,8],i:[0,1,3,4,5,8,9,10,11],id:[0,3,10],identifi:8,idf:0,ieee:0,ignor:[8,10,11],ii:8,iid:[1,5,6],illustr:[3,4,5],imdb:[0,5,10],implement:[0,1,3,4,5,6,8,9,10,11],implicit:8,impos:[4,8],improv:[3,8,9,11],includ:[0,1,3,5,6,10,11],inconveni:8,inde:[3,4],independ:[8,11],index:[0,3,6,8,9,10,11],indextransform:10,indic:[0,1,3,4,5,8,10,11],individu:[1,3],infer:[0,10],inform:[0,1,3,4,8,10,11],infrequ:10,inherit:3,init:3,init_hidden:[],initi:[0,9],inplac:[1,3,10,11],input:[3,5,8,9,11],insight:5,inspir:3,instal:[0,3,6,9,11],instanc:[0,3,4,5,6,8,9,10,11],instanti:[0,1,3,4,9,11],instead:[1,3,4,11],integ:[3,8,9,10,11],integr:6,interest:[1,5,6,8,10],interestingli:5,interfac:[0,1,11],intern:[0,3,10],interpret:[5,6,11],interv:[1,5,8,10],introduc:1,invok:[0,1,3,8,10],involv:[2,5,8],io:[],ionospher:0,iri:0,irrespect:[5,11],isaggreg:11,isbinari:[8,10,11],isomer:8,isometr:[5,8],isprobabilist:11,isti:[],item:8,iter:[0,8,11],its:[3,4,8,9,11],itself:[3,8,11],j:[0,3,10,11],joachim:[3,9,11],job:[2,8],joblib:2,join:8,just:[1,3],k:[3,6,8,10,11],keep:8,kei:[8,10],kept:10,kernel:9,kernel_height:9,keyword:[10,11],kfcv:[0,10,11],kindl:[0,1,3,5,10,11],kl:8,kld:[1,2,8,9,11],know:3,knowledg:[0,3,10],known:[0,3,4,11],kraemer:8,kullback:[1,3,8,11],kwarg:[9,10,11],l1:[8,11],l:11,label:[0,3,4,5,6,8,9,10,11],labelledcollect:[0,3,4,8,10,11],larg:4,larger:[10,11],largest:8,last:[1,3,5,8,9,10],lastli:3,latex:5,latinn:[3,11],latter:11,layer:[3,9,11],lazi:11,lead:[1,10],learn:[1,2,3,4,6,8,9,10,11],learner:[3,4,9,11],least:[0,10],leav:10,left:10,legend:8,leibler:[1,3,8,11],len:8,length:[9,10],less:[8,10],let:[1,3],level:[],leverag:3,leyend:8,like:[0,1,3,5,8,9,10,11],likelihood:11,limit:[5,8,10,11],line:[1,3,8],linear:[5,11],linear_model:[1,3,4,6,9],linearsvc:[3,5,10],link:[],linspac:5,list:[0,5,8,9,10,11],listedcolormap:8,literatur:[0,1,4,6],load:[0,3,8,10,11],loader:[0,10],loader_func:[0,10],loader_kwarg:10,local:8,log:[8,10],logist:[1,3,9,11],logisticregress:[1,3,4,6,9,11],logscal:8,logspac:[4,11],longer:8,longest:9,look:[0,1,3,5,11],loop:11,loss:[6,9,11],low:[5,8,9],lower:[5,8,11],lower_is_bett:8,lowest:5,lowranklogisticregress:9,lr:[1,3,9,11],lstm:[3,9,11],lstm_class_nlay:9,lstm_hidden_s:11,lstm_nlayer:11,lstmnet:9,m:[3,8,11],machin:[1,4,6],macro:8,made:[0,2,8,10,11],mae:[1,4,6,8,9,11],mae_loss:11,mai:8,main:5,maintain:[3,11],make:[0,1,3,11],makedir:8,mammograph:0,manag:[0,3,10],mani:[1,3,4,5,6,8,10,11],manner:0,manual:0,map:[1,9],map_parallel:8,margin:9,mass:8,math:[],mathcal:8,matplotlib:[2,8],matric:[0,5,10],matrix:[5,8,11],max:11,max_it:11,max_sample_s:11,maxim:[6,11],maximum:[1,8,9,11],maximumlikelihoodprevalenceestim:11,md:[],mean:[0,1,3,4,5,6,8,9,10,11],mean_absolute_error:8,mean_relative_absolute_error:8,measur:[2,3,4,5,6,8,11],median:11,mediansweep2:11,mediansweep:11,member:[3,11],memori:9,mention:3,merg:5,met:10,meta:[6,7,8],meth:[],method:[0,1,4,5,6,7,8],method_data:5,method_nam:[5,8],method_ord:8,metric:[1,3,4,6,8,11],might:[1,8,10],min_df:[1,3,4,5,10,11],min_po:11,mine:[0,3],minim:[8,11],minimum:[10,11],minimun:10,mining6:10,minu:8,misclassif:11,miss:8,mixtur:[3,11],mkld:[1,8,11],ml:10,mlpe:11,mnkld:[1,8,11],mock:[8,9],modal:4,model:[0,1,5,6,8,9,11],model_select:[4,7,11],modifi:[3,8],modul:[0,1,3,5,6,7],moment:[0,3],monitor:8,more:[3,5,8,11],moreo:[0,3,4,10,11],most:[0,3,5,6,8,10,11],movi:0,mrae:[1,6,8,9,11],ms2:11,ms:11,mse:[1,3,6,8,11],msg:[],multiclass:8,multipli:8,multiprocess:8,multivari:[3,9],must:[3,10,11],mutual:11,my:[],my_arrai:8,my_collect:10,my_custom_load:0,my_data:0,mycustomloss:3,n:[0,1,8,9,11],n_bin:[5,8],n_class:[1,3,8,9,10,11],n_classes_:11,n_compon:9,n_dimens:9,n_epoch:11,n_featur:9,n_instanc:[8,9,11],n_job:[1,3,4,8,10,11],n_preval:[0,8,10],n_prevpoint:[1,4,5,8],n_repeat:[1,8],n_repetit:[1,4,5,8],n_sampl:[8,9],name:[5,8,9,10,11],nativ:6,natur:[1,8,10,11],natural_prevalence_predict:8,natural_prevalence_protocol:8,natural_prevalence_report:8,natural_sampling_gener:10,natural_sampling_index_gener:10,nbin:[5,8],ndarrai:[1,3,8,10,11],necessarili:[],need:[0,3,8,10,11],neg:[0,5,8,11],nest:[],net:9,network:[0,8,9,10,11],neural:[0,7,8,10],neuralclassifiertrain:[3,9,11],neutral:0,next:[4,8,9,10],nfold:[0,10],nkld:[1,2,6,8,9,11],nn:[9,11],nogap:10,non:3,non_aggreg:[7,8],none:[1,4,8,9,10,11],nonetheless:4,nor:3,normal:[0,1,3,8,10,11],normalize_preval:8,note:[1,3,4,5,8,10],noth:11,now:5,nowadai:3,np:[1,3,4,5,8,10,11],npp:[8,10],nprevpoint:[],nrepeat:[0,10],num_prevalence_combin:[1,8],number:[0,1,3,5,8,9,10,11],numer:[0,1,3,6,10,11],numpi:[2,4,8,9,11],o_l6x_pcf09mdetq4tu7jk98mxfbgsxp9zso14jkuiyudgfg0:[],object:[0,8,9,10,11],observ:1,obtain:[1,4,8,11],obtaind:8,obvious:8,occur:[5,10],occurr:10,octob:[0,3],off:9,offer:[3,6],older:2,omd:[0,10],ommit:[1,8],onc:[1,3,5,8],one:[0,1,3,4,5,8,10,11],ones:[1,3,5,8,10],onevsal:[3,11],onli:[0,3,5,8,9,10,11],open:[0,6,10],oper:3,opt:4,optim:[2,3,4,8,9,11],optimize_threshold:[],option:[0,1,3,5,8,10,11],order:[0,2,3,5,8,10,11],order_bi:11,org:10,orient:[3,6,8,11],origin:[0,3,10],os:[0,8],other:[1,3,5,6,8,10,11],otherwis:[0,3,8,10,11],our:[],out:[3,4,5,8,9,10,11],outcom:5,outer:8,outlier:8,output:[0,1,3,4,8,9,10,11],outsid:11,over:[3,4,8],overal:1,overestim:5,overrid:3,overridden:[3,11],own:4,p:[0,3,8,10,11],p_hat:8,p_i:8,pacc:[1,3,5,8,11],packag:[0,2,3,6,7],pad:[9,10],pad_length:9,padding_length:9,page:[0,2,6],pageblock:0,pair:[0,8,11],panda:[1,2,8],paper:[0,3],parallel:[1,3,8,10,11],param:[4,9,11],param_grid:[4,8,11],param_mod_sel:11,param_model_sel:11,paramet:[1,3,4,8,9,10,11],parent:8,part:[3,10],particular:[0,1,3],particularli:1,pass:[0,1,5,8,9,11],past:1,patch:[2,3,9,11],path:[0,3,5,8,9,10,11],patienc:[8,9,11],pattern:3,pca:[],pcalr:[],pcc:[3,4,5,11],pd:1,pdf:5,peopl:[],percentil:8,perf:[6,9,11],perform:[1,3,4,5,6,8,9,11],perman:8,phase:11,phonem:0,pick:4,pickl:[3,8,10,11],pickle_path:8,pickled_resourc:8,pii:[],pip:2,pipelin:[],pkl:8,plai:0,plan:3,pleas:3,plot:[6,7],png:5,point:[0,1,3,8,10],polici:[3,11],popular:6,portion:4,pos_class:[8,10],posit:[0,3,5,8,10,11],possibl:[1,3,8],post:8,posterior:[3,8,9,11],posterior_prob:[3,11],postpon:3,potter:0,pp:[0,3],pprox:[],practic:[0,4],pre:[0,3],prec:[0,8],preced:10,precis:[0,1,8],preclassifi:3,predefin:10,predict:[3,4,5,8,9,11],predict_proba:[3,9,11],predictor:1,prefer:8,preliminari:11,prepare_svmperf:[2,3],preprint:4,preprocess:[0,1,3,7,8,11],present:[0,3,10],preserv:[1,5,8,10],pretti:5,prev:[0,1,8,10],prevail:3,preval:[0,1,3,4,5,6,8,10,11],prevalence_estim:8,prevalence_from_label:8,prevalence_from_prob:8,prevalence_linspac:8,prevel:11,previou:3,previous:[],prevs_estim:11,prevs_hat:[1,8],princip:9,print:[0,1,3,4,6,9,10],prior:[1,3,4,5,6,8,11],priori:3,probabilist:[3,11],probabilisticadjustedclassifyandcount:11,probabilisticclassifyandcount:11,probabl:[1,3,4,5,6,8,9,11],problem:[0,3,5,8,10,11],procedur:[3,6],proceed:[0,3,10],process:[3,4,8],processor:3,procol:1,produc:[0,1,5,8],product:3,progress:[8,10],properli:0,properti:[3,8,9,10,11],proport:[3,4,8,9,10,11],propos:[2,3,11],protocl:8,protocol:[0,3,4,5,6,8,10,11],provid:[0,3,5,6,11],ptecondestim:11,ptr:[3,11],ptr_polici:[],purpos:[0,11],put:11,python:[0,6],pytorch:[2,11],q:[0,2,3,8,9,11],q_i:8,qacc:9,qdrop_p:11,qf1:9,qgm:9,qp:[0,1,3,4,5,6,8,10,11],quanet:[2,6,9,11],quanetmodul:11,quanettrain:11,quantif:[0,1,6,8,9,10,11],quantifi:[3,4,5,6,8,11],quantification_error:8,quantiti:8,quapi:[0,1,2,3,4,5],quapy_data:0,quay_data:10,question:8,quevedo:[0,3,10],quick:[],quit:8,r:[0,3,8,10],rac:[],rae:[1,2,8,11],rais:[3,8,11],rand:8,random:[1,3,4,5,8,10],random_se:[1,8],random_st:10,randomli:0,rang:[0,5,8,11],rank:[3,9],rare:10,rate:[3,8,9,11],rather:[1,4],raw:10,rb:0,re:[3,4,10],reach:11,read:10,reader:[7,8],readm:[],real:[8,9,10,11],reason:[3,5,6],recal:8,receiv:[0,3,5],recip:11,recognit:3,recommend:[1,5,11],recomput:11,recurr:[0,3,10],recurs:11,red:0,red_siz:[3,11],reduc:[0,10],reduce_column:[0,10],refer:[9,10],refit:[4,8],regard:4,regardless:10,regim:8,region:8,regist:11,regress:9,regressor:[1,3],reindex_label:10,reiniti:9,rel:[1,3,8,10,11],relative_absolute_error:8,reli:[1,3,11],reliabl:3,rememb:5,remov:[10,11],repeat:[8,10],repetit:8,repl:[],replac:[0,3,10],replic:[1,4,8],report:[1,8],repositori:[0,10],repr_siz:9,repres:[1,3,5,8,10,11],represent:[0,3,8,9,11],reproduc:10,request:[0,8,10],requir:[0,1,3,6,9],reset_net_param:9,resourc:8,resp:11,respect:[0,1,5,8,11],respond:3,rest:[8,10,11],result:[1,2,3,4,5,6,8,11],retain:[0,3,9,11],retrain:4,return_constrained_dim:8,reus:[0,3,8],review:[5,6,10],reviews_sentiment_dataset:[0,10],rewrit:5,right:[4,8,10],role:0,root:6,roughli:0,round:10,routin:[8,10,11],row:[8,10],run:[0,1,2,3,4,5,8,10,11],s003132031400291x:[],s10618:[],s:[0,1,3,4,5,8,9,10,11],saeren:[3,11],sai:[],said:3,same:[0,3,5,8,10,11],sampl:[0,1,3,4,5,6,8,9,10,11],sample_s:[0,1,3,4,5,8,10,11],sampling_from_index:[0,10],sampling_index:[0,10],sander:[0,10],save:[5,8],save_or_show:[],save_text_fil:8,savepath:[5,8],scale:8,scall:10,scenario:[1,3,4,5,6],scienc:3,sciencedirect:[],scikit:[2,3,4,10],scipi:[2,10],score:[0,1,4,8,9,10],script:[1,2,3,6,11],se:[1,8],search:[3,4,6,8],sebastiani:[0,3,4,10,11],second:[0,1,3,5,8,10],secondari:8,section:4,see:[0,1,2,3,4,5,6,8,9,10,11],seed:[1,4,8],seem:3,seemingli:5,seen:[5,8,11],select:[0,3,6,8,10,11],selector:3,self:[3,8,9,10,11],semeion:0,semev:0,semeval13:[0,10],semeval14:[0,10],semeval15:[0,10],semeval16:[0,6,10],sentenc:10,sentiment:[3,6,10],separ:[8,10],sequenc:8,seri:0,serv:3,set:[0,1,3,4,5,6,8,9,10,11],set_opt:1,set_param:[3,8,9,11],set_siz:[],sever:0,sh:[2,3],shape:[5,8,9,10,11],share:[0,10],shift:[1,4,6,8,11],shorter:9,shoud:3,should:[0,1,3,4,5,6,9,10,11],show:[0,1,3,4,5,8,9,10,11],show_dens:8,show_std:[5,8],showcas:5,shown:[1,5,8],shuffl:[9,10],side:8,sign:8,signific:1,significantli:8,silent:[8,11],simeq:[],similar:[8,11],simpl:[0,3,5,11],simplest:3,simplex:[0,8],simpli:[1,2,3,4,5,6,8,11],sinc:[0,1,3,5,8,10,11],singl:[1,3,6,11],size:[0,1,3,8,9,10,11],sklearn:[1,3,4,5,6,9,10,11],sld:[3,11],slice:8,smooth:[1,8],smooth_limits_epsilon:8,so:[0,1,3,5,8,9,10,11],social:[0,3,10],soft:3,softwar:0,solid:5,solut:8,solv:[4,11],solve_adjust:11,some:[0,1,3,5,8,10,11],some_arrai:8,sometim:1,sonar:0,sort:11,sourc:[2,3,6,9],sout:[],space:[0,4,8,9],spambas:0,spars:[0,10],special:[0,5,10],specif:[3,4],specifi:[0,1,3,5,8,9,10],spectf:0,spectrum:[0,1,4,5,8],speed:[3,11],split:[0,3,4,5,8,9,10,11],split_stratifi:10,splitstratifi:10,spmatrix:10,springer:[],sqrt:8,squar:[1,3,8],sst:[0,10],stabil:[1,11],stabl:10,stackexchang:8,stand:[8,11],standard:[0,1,5,8,10,11],star:8,start:4,stat:10,state:8,statist:[0,1,8,11],stats_siz:11,std:9,stdout:8,step:[5,8],stop:[8,9,11],store:[0,9,10,11],str:[0,8,10],strategi:[3,4],stratif:10,stratifi:[0,3,10,11],stride:9,string:[1,8,10,11],strongli:[4,5],strprev:[0,1,8],structur:[3,11],studi:[0,3,10],style:10,subclass:11,subdir:8,subinterv:5,sublinear_tf:10,submit:0,submodul:7,subobject:[],suboptim:4,subpackag:7,subsequ:10,subtract:[0,8,10],subtyp:10,suffic:5,suffici:[],sum:[8,11],sum_:8,summar:0,supervis:[4,6],support:[3,6,9,10],surfac:10,surpass:1,svm:[3,5,6,9,10,11],svm_light:[],svm_perf:[],svm_perf_classifi:9,svm_perf_learn:9,svm_perf_quantif:[2,3],svmae:[3,11],svmkld:[3,11],svmnkld:[3,11],svmperf:[2,3,7,8,11],svmperf_bas:[9,11],svmperf_hom:3,svmq:[3,11],svmrae:[3,11],sweep:11,syntax:5,system:[4,11],t50:11,t:[0,1,3,8],tab10:8,tail:8,tail_density_threshold:8,take:[0,3,5,8,10,11],taken:[3,8,9,10],target:[3,5,6,8,9,11],task:[3,4,10],te:[8,10],temp_se:8,tempor:8,tend:5,tendenc:5,tensor:9,term:[0,1,3,4,5,6,8,9,10,11],test:[0,1,3,4,5,6,8,9,10,11],test_bas:[],test_dataset:[],test_method:[],test_path:[0,10],test_sampl:8,test_split:10,text2tfidf:[0,1,3,10],text:[0,3,8,9,10,11],textclassifiernet:9,textual:[0,6,10],tf:[0,10],tfidf:[0,4,5,10],tfidfvector:10,than:[1,4,5,8,9,10],thei:[0,3,11],them:[0,3,11],theoret:4,thereaft:1,therefor:[8,10],thi:[0,1,2,3,4,5,6,8,9,10,11],thing:3,third:[1,5],thorsten:9,those:[1,3,4,5,8,9,11],though:[3,8],three:[0,5],threshold:[8,11],thresholdoptim:11,through:[3,8],thu:[3,4,5,8,11],tictacto:0,time:[0,1,3,8,10],timeout:8,timeouterror:8,timer:8,titl:8,tj:[],tn:8,token:[0,9,10],tool:[1,6],top:[3,8,11],torch:[3,9,11],torchdataset:9,total:8,toward:[5,10],tp:8,tpr:[8,11],tqdm:2,tr:10,tr_iter_per_poch:11,tr_prev:[5,8,11],track:8,trade:9,tradition:1,train:[0,1,3,4,5,6,8,9,10,11],train_path:[0,10],train_prev:[5,8],train_prop:10,train_siz:10,train_val_split:[],trainer:9,training_help:[],training_preval:5,training_s:5,transact:3,transform:[0,9,10,11],transfus:0,trivial:3,true_prev:[1,5,8],true_preval:6,truncatedsvd:9,truth:11,ttest_alpha:8,tupl:[8,10,11],turn:4,tweet:[0,3,10],twitter:[6,10],twitter_sentiment_datasets_test:[0,10],twitter_sentiment_datasets_train:[0,10],two:[0,1,3,4,5,8,10,11],txt:8,type:[0,3,8,10,11],typic:[1,4,5,8,9,10,11],u1:10,uci:[6,10],uci_dataset:10,unabl:0,unadjust:5,unalt:9,unbias:5,uncompress:0,under:1,underestim:5,underlin:8,understand:8,unfortun:5,unifi:[0,11],uniform:[8,10],uniform_prevalence_sampl:8,uniform_sampl:10,uniform_sampling_index:10,uniform_simplex_sampl:8,uniformli:[8,10],union:[8,11],uniqu:10,unit:[0,8],unix:0,unk:10,unknown:10,unlabel:11,unless:11,unlik:[1,4],until:11,unus:[8,9],up:[3,4,8,9,11],updat:11,url:8,us:[0,1,3,4,5,6,8,9,10,11],user:[0,1,5],utf:10,util:[7,9],v:3,va_iter_per_poch:11,val:[0,10],val_split:[3,4,8,9,11],valid:[0,1,3,4,5,8,9,10,11],valid_loss:[3,9,11],valid_polici:11,valu:[0,1,3,8,9,10,11],variabl:[1,3,5,8,10],varianc:[0,5],variant:[5,6,11],varieti:4,variou:[1,5],vector:[0,8,9,10],verbos:[0,1,4,8,9,10,11],veri:[3,5],versatil:6,version:[2,9,11],vertic:8,vertical_xtick:8,via:[0,2,3,11],view:5,visual:[5,6],vline:8,vocab_s:9,vocabulari:[9,10],vocabulary_s:[3,9,10,11],vs:[3,8],w:[0,3,10],wa:[0,3,5,8,10,11],wai:[1,11],wait:9,want:[3,4],warn:10,wb:[0,10],wdbc:0,we:[0,1,3,4,5,6],weight:[9,10],weight_decai:9,well:[0,3,4,5,11],were:0,what:3,whcih:10,when:[0,1,3,4,5,8,9,10],whenev:[5,8],where:[3,5,8,9,10,11],wherebi:4,whether:[8,9,10,11],which:[0,1,3,4,5,8,9,10,11],white:0,whole:[0,1,3,4,8],whose:[10,11],why:3,wide:5,wiki:[0,3],wine:0,within:[8,11],without:[1,3,8,10],word:[1,3,6,9,10,11],work:[1,3,4,5,10],worker:[1,8,10,11],wors:[4,5,8],would:[0,1,3,5,6,8,10,11],wrapper:[8,9,10,11],written:6,www:[],x2:10,x:[5,8,9,10,11],x_error:8,xavier:9,xavier_uniform:9,xlrd:[0,2],xy:10,y:[5,8,9,10,11],y_:[],y_error:8,y_i:11,y_j:11,y_pred:8,y_true:8,ye:[],yeast:[0,10],yield:[5,8,10,11],yin:[],you:[2,3],your:3,z:[0,10],zero:[0,8],zfthyovrzwxmgfzylqw_y8cagg:[],zip:[0,5]},titles:["Datasets","Evaluation","Installation","Quantification Methods","Model Selection","Plotting","Welcome to QuaPy\u2019s documentation!","quapy","quapy package","quapy.classification package","quapy.data package","quapy.method package"],titleterms:{"function":8,A:6,The:3,ad:0,aggreg:[3,11],base:[10,11],bia:5,classif:[4,9],classifi:3,content:[6,8,9,10,11],count:3,custom:0,data:[0,10],dataset:[0,10],diagon:5,distanc:3,document:6,drift:5,emq:3,ensembl:3,error:[1,5,8],evalu:[1,8],ex:[],exampl:6,expect:3,explicit:3,featur:6,get:[],hdy:3,helling:3,indic:6,instal:2,introduct:6,issu:0,learn:0,loss:[2,3,4],machin:0,maxim:3,measur:1,meta:[3,11],method:[3,9,11],minim:3,model:[3,4],model_select:8,modul:[8,9,10,11],network:3,neural:[3,9,11],non_aggreg:11,orient:[2,4],packag:[8,9,10,11],perf:2,plot:[5,8],preprocess:10,process:0,protocol:1,quanet:3,quantif:[2,3,4,5],quapi:[6,7,8,9,10,11],quick:6,reader:10,readm:[],requir:2,review:0,s:6,select:4,sentiment:0,start:[],submodul:[8,9,10,11],subpackag:8,svm:2,svmperf:9,tabl:6,target:4,test:[],test_bas:[],test_dataset:[],test_method:[],titl:[],twitter:0,uci:0,util:8,variant:3,welcom:6,y:3}})