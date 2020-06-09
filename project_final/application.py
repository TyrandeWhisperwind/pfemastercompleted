import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSlot, QObject, pyqtSignal
from window import Ui_MainWindow
import pickle
import os

from dataset import DataSet, load_data, dump_array
from preprocess import mean, svd_uu_100k, gnb_uu_100k
from fc import pearson_distance
from semantic import semantic, jaccard_distance, wp_distance
from hybrid import alpha_beta_hybrid, semantic_based_fc_hybrid, fc_based_semantic_hybrid
from fc_user_user import fc_user_user
from fc_item_item import fc_item_item

from surprise import SVD, Dataset, Reader

from clustering.knn import KNN, KNNMultiview, createDictTestMovies
from clustering.kmedoids import kmedoids
from clustering.MAE_RMSE import MAE_RMSE, createDictTestMovies as createDictTestMovies2

from bso import BSO
from fc_kmedoids import bso_eval

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.tasks = QThreadPool()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        ############Preprocessing#########################################
        self.ui.base_data_btn.clicked.connect(self.on_base_btn_clicked)
        self.ui.user_data_btn.clicked.connect(self.on_user_btn_clicked)
        self.ui.item_data_btn.clicked.connect(self.on_item_btn_clicked)
        self.ui.load_data_btn.clicked.connect(self.on_load_data)
        self.ui.process_filtering_technique_btn.clicked.connect(self.on_process_filter)
        self.ui.preprocess_usage_matrix_btn.clicked.connect(self.on_preprocess_usage_matrix)
        self.ui.save_dataset_btn.clicked.connect(self.on_save_dataset)
        self.ui.prediction_model_btn.clicked.connect(lambda : self.ui.prediction_model_path.setText(self.on_load_dataset()))
        
        ###########SVD MODEL#############################################
        self.ui.train_data_load_btn.clicked.connect(self.on_load_train_data)
        self.ui.svd_model_train_btn.clicked.connect(self.on_create_svd)
        self.ui.save_svd_model_btn.clicked.connect(self.on_save_svd_model)
        ###########Filtering#############################################
        self.ui.f_dataset_load_btn.clicked.connect(lambda :self.ui.f_dataset_path.setText(self.on_load_dataset()))
        self.ui.save_distance_matrix_btn.clicked.connect(self.on_save_dist_mat)
        self.ui.sem_dist_matrix_load_btn.clicked.connect(lambda :self.ui.sem_dist_matrix_path.setText(self.on_load_dist_mat()))
        self.ui.fc_dist_matrix_load_btn.clicked.connect(lambda :self.ui.fc_dist_matrix_path.setText(self.on_load_dist_mat()))
        self.ui.taxonomy_btn.clicked.connect(lambda :self.ui.taxonomy_path.setText(self.on_load_taxonomy()))

        ##########Test Filtering#########################################
        self.ui.uu_radio_btn.toggled.connect(self.on_t_r_b_clicked)
        self.ui.ii_radio_btn.toggled.connect(self.on_t_r_b_clicked)
        self.ui.t_dataset_btn.clicked.connect(lambda :self.ui.t_dataset_path.setText(self.on_load_dataset()))
        self.ui.t_dist_mat_btn.clicked.connect(lambda :self.ui.t_dist_mat_path.setText(self.on_load_dist_mat()))
        self.ui.t_test_file_btn.clicked.connect(lambda :self.ui.t_test_file_path.setText(self.on_load_test_file()))
        self.ui.t_test_btn.clicked.connect(self.on_t_test_clicked)

        ##########Test Clustering and Optimisation#######################
        self.ui.c_dataset_btn.clicked.connect(lambda :self.ui.c_dataset_path.setText(self.on_load_dataset()))
        self.ui.c_dist_mat_btn_1.clicked.connect(lambda :self.ui.c_dist_mat_path_1.setText(self.on_load_dist_mat()))
        self.ui.c_dist_mat_btn_2.clicked.connect(lambda :self.ui.c_dist_mat_path_2.setText(self.on_load_dist_mat()))
        self.ui.c_test_file_btn.clicked.connect(lambda :self.ui.c_test_file_path.setText(self.on_load_test_file()))
        self.ui.knn_run_btn.clicked.connect(self.run_knn)
        self.ui.kmedoids_run_btn.clicked.connect(self.run_kmedoids)

        self.ui.filtering_methods_cb.addItems(
            [
                "Colabortive filtering",
                "Semantic filtering",
                "Weighted Hybrid filtering",
                "Semantic based Collabrative filtering",
                "Collabrative based semantic filtering"
            ]
        )

        self.ui.preprocessing_cb.addItems(
            [
                "None",
                "Mean rating of user",
                "Mean rating of item",
                "SVD model prediction"
            ]
        )

        
    
    def on_base_btn_clicked(self):
        self.ui.base_data_path.setText(
            QFileDialog.getOpenFileName(self, "Base Dataset File", "/home/imad", "base (*.base)")[0]
        )
        print('[input file]', self.ui.base_data_path.text())
    
    def on_item_btn_clicked(self):
        self.ui.item_data_path.setText(
            QFileDialog.getOpenFileName(self, "item File", "/home/imad", "item (*.item)")[0]
        )
        print('[input file]', self.ui.item_data_path.text())

    def on_user_btn_clicked(self):
        self.ui.user_data_path.setText(
            QFileDialog.getOpenFileName(self, "Base Dataset File", "/home/imad", "user (*.user)")[0]
        )
        print('[input file]', self.ui.user_data_path.text())

    def on_load_data(self):
        self.ui.status_label.setText('Loading data...')
        
        w = Worker(self.load_data)
        w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
        self.tasks.start(w)

    def load_data(self):
        print('[load data] started')
        self.dataset = DataSet(
            self.ui.base_data_path.text(),
            self.ui.user_data_path.text(),
            self.ui.item_data_path.text()
        )
        print('[load data] finished')
        #print(self.dataset)

    def on_preprocess_usage_matrix(self):
        if self.dataset is None:
            print("Import a dataset")
            return
        self.ui.status_label.setText('Preprocessing Data...')
        w = Worker(self.preprocess_usage_matrix)
        w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
        self.tasks.start(w)

    def preprocess_usage_matrix(self):
        if self.ui.preprocessing_cb.currentIndex() == 0:
            return
        elif self.ui.preprocessing_cb.currentIndex() == 1:
            self.dataset.set_usage_matrix(mean(self.dataset.get_usage_matrix()))
        elif self.ui.preprocessing_cb.currentIndex() == 2:
            self.dataset.set_usage_matrix(mean(self.dataset.get_usage_matrix(), axis=0))
        elif self.ui.preprocessing_cb.currentIndex() == 3:
            if self.ui.prediction_model_path.text() == "":
                return
            self.dataset.set_usage_matrix(
                svd_uu_100k(
                    self.dataset.get_usage_matrix(),
                    self.ui.prediction_model_path.text()
                )
            )
        elif self.ui.preprocessing_cb.currentIndex() == 4:
            self.dataset.set_usage_matrix(gnb_uu_100k(self.dataset.get_usage_matrix()))

    def on_save_dataset(self):
        if self.dataset is None:
            print("Import a dataset")
            return
        
        filename = QFileDialog.getSaveFileName()[0]
        print(filename)
        if filename is None: return
        self.ui.status_label.setText('Saving Dataset...')
        pickle.dump(self.dataset, open(filename, "wb"))
        self.ui.status_label.setText('Done')

    ############################################################################

    def on_load_train_data(self):
        self.ui.train_data_path.setText(
            QFileDialog.getOpenFileName(self, "Training data File", "/home/imad", "base (*.base)")[0]
        )
    
    def on_create_svd(self):
        if self.ui.train_data_path.text() == '':
            return
        w = Worker(self.create_svd)
        w.signals.result.connect(self.set_model)
        w.signals.finished.connect(lambda : self.ui.status_label.setText('Done.'))
        self.ui.status_label.setText('Creating an SVD Model...')
        self.tasks.start(w)


    def create_svd(self):
        df = pd.read_csv(self.ui.train_data_path.text(),sep='\t',names=['user','item','rating'])
        #print(df)
        data = Dataset.load_from_df(df, Reader(rating_scale=(1,5)))
        #data = Dataset.load_builtin('ml-100k')
        svd_model = SVD()
        svd_model.fit(data.build_full_trainset())
        #cross_validate(svd_model, data, cv=5, verbose=True)

        return svd_model

    def set_model(self, model):
        self.model = model

    def on_save_svd_model(self):
        if self.model is None:
            print("No model created to be saved")
            return
        filename = QFileDialog.getSaveFileName()[0]
        print(filename)
        if filename is None: return
        self.ui.status_label.setText('Saving SVD Model...')
        pickle.dump(self.model, open(filename, "wb"))
        self.ui.status_label.setText('Done')
    ############################################################################

    def on_load_dataset(self):
        return QFileDialog.getOpenFileName(self, "SAV File", "/home/imad", "sav (*.sav)")[0]

    def on_load_taxonomy(self):
        return QFileDialog.getOpenFileName(self, "Taxonomy File", "./", "category (*.category)")[0]

    def on_process_filter(self):
        if self.ui.filtering_methods_cb.currentIndex() == 0:
            if self.ui.f_dataset_path.text() != "" and os.path.isfile(self.ui.f_dataset_path.text()):
                try:
                    self.dataset = pickle.load(open(self.ui.f_dataset_path.text(), "rb"))
                except:
                    return
            else:
                return
            print(self.dataset.get_usage_matrix())
            self.ui.status_label.setText('Running CF user/user...')
            w = Worker(pearson_distance, self.dataset.get_usage_matrix())
            w.signals.result.connect(self.set_dist_mat)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            self.tasks.start(w)

        elif self.ui.filtering_methods_cb.currentIndex() == 1:
            if self.ui.f_dataset_path.text() != "" and os.path.isfile(self.ui.f_dataset_path.text()):
                try:
                    self.dataset = pickle.load(open(self.ui.f_dataset_path.text(), "rb"))
                except:
                    return
            else:
                return
            w = None
            if self.ui.jaccard_rb.isChecked():
                self.ui.status_label.setText('Running semantic filtering user/user with Jaccard ...')
                w = Worker(
                    semantic,
                    self.dataset.get_usage_matrix(),
                    self.dataset.get_movie_matrix(),
                    jaccard_distance
                )
            elif self.ui.taxonomy_path.text() != "":
                self.ui.status_label.setText('Running semantic filtering user/user with WP ...')
                taxonomy = load_data(self.ui.taxonomy_path.text(), sep='\t')
                taxonomy = taxonomy[:, 1]
                taxonomy = np.insert(taxonomy, 0, -1)
                print(taxonomy)
                w = Worker(
                    semantic,
                    self.dataset.get_usage_matrix(),
                    self.dataset.get_movie_matrix(),
                    wp_distance,
                    taxonomy
                )
            #print(self.dataset.get_usage_matrix())
            if w is None:
                self.ui.status_label.setText('Error: missing parameters.')
                return
            w.signals.result.connect(self.set_dist_mat)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            self.tasks.start(w)
        elif self.ui.filtering_methods_cb.currentIndex() == 2:
            sem_dist_mat = None; fc_dist_mat = None
            if self.ui.fc_dist_matrix_path.text() != "" and os.path.isfile(self.ui.fc_dist_matrix_path.text()):
                try:
                    fc_dist_mat = load_data(self.ui.fc_dist_matrix_path.text())
                except:
                    return
            else:
                return
            
            if self.ui.sem_dist_matrix_path.text() != "" and os.path.isfile(self.ui.sem_dist_matrix_path.text()):
                try:
                    sem_dist_mat = load_data(self.ui.sem_dist_matrix_path.text())
                except:
                    return
            else:
                return
            
            if self.ui.alpha_edit.text() == "": return

            #print(self.dataset.get_usage_matrix())
            self.ui.status_label.setText('Running weighted hybrid user/user...')
            
            w = Worker(
                alpha_beta_hybrid,
                fc_dist_mat,
                sem_dist_mat,
                float(self.ui.alpha_edit.text())
            )
            w.signals.result.connect(self.set_dist_mat)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            self.tasks.start(w)
        elif self.ui.filtering_methods_cb.currentIndex() == 3:
            sem_dist_mat = None; fc_dist_mat = None
            if self.ui.fc_dist_matrix_path.text() != "" and os.path.isfile(self.ui.fc_dist_matrix_path.text()):
                try:
                    fc_dist_mat = load_data(self.ui.fc_dist_matrix_path.text())
                except:
                    return
            else:
                return
            
            if self.ui.sem_dist_matrix_path.text() != "" and os.path.isfile(self.ui.sem_dist_matrix_path.text()):
                try:
                    sem_dist_mat = load_data(self.ui.sem_dist_matrix_path.text())
                except:
                    return
            else:
                return
            
            #if self.ui.alpha_edit.text() == "": return

            #print(self.dataset.get_usage_matrix())
            self.ui.status_label.setText('Running semantic based FC user/user...')
            
            w = Worker(
                semantic_based_fc_hybrid,
                fc_dist_mat,
                sem_dist_mat
            )
            w.signals.result.connect(self.set_dist_mat)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            self.tasks.start(w)
        elif self.ui.filtering_methods_cb.currentIndex() == 4:
            fc_dist_mat = None
            if self.ui.fc_dist_matrix_path.text() != "" and os.path.isfile(self.ui.fc_dist_matrix_path.text()):
                try:
                    fc_dist_mat = load_data(self.ui.fc_dist_matrix_path.text())
                except:
                    return
            else:
                return
            
            if self.ui.f_dataset_path.text() != "" and os.path.isfile(self.ui.f_dataset_path.text()):
                try:
                    self.dataset = pickle.load(open(self.ui.f_dataset_path.text(), "rb"))
                except:
                    return
            else:
                return
            
            #if self.ui.alpha_edit.text() == "": return

            #print(self.dataset.get_usage_matrix())
            self.ui.status_label.setText('Running FC based semantic user/user...')
            
            w = Worker(
                self.run_fc_based_sem_hyb,
                fc_dist_mat,
                self.dataset.get_usage_matrix()
            )
            w.signals.result.connect(self.set_dist_mat)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            self.tasks.start(w)

    def run_fc_based_sem_hyb(self, fc_dist_mat, usage_matrix):
        item_mat = np.array(
            [
                [
                    (i, row[i])
                    for i in range(len(row))
                ]
                for row in fc_dist_mat
            ]
        )
        return fc_based_semantic_hybrid(item_mat, usage_matrix)

    def on_save_dist_mat(self):
        if self.dist_mat is None:
            print("No data provided")
            return
        
        filename = QFileDialog.getSaveFileName()
        print(filename)
        if filename is None: return
        self.ui.status_label.setText('Saving Distance Matrix...')
        dump_array(self.dist_mat, filename[0])
        self.ui.status_label.setText('Done')

    def set_dist_mat(self, m):
        print(
            self.ui.filtering_methods_cb.currentText(),
            "\n",
            m
        )
        self.dist_mat = m
    

    ##########################################################################

    def on_t_r_b_clicked(self):
        pass

    def on_load_dist_mat(self):
        return QFileDialog.getOpenFileName(self, "Distance Matrix", "/home/imad", "csv (*.csv)")[0]
    
    def on_load_test_file(self):
        return QFileDialog.getOpenFileName(self, "Distance Matrix", "/home/imad", "test (*.test)")[0]
    
    def on_t_test_clicked(self):
        if not os.path.isfile(self.ui.t_test_file_path.text()):
            return
        if not os.path.isfile(self.ui.t_dataset_path.text()):
            return
        if not os.path.isfile(self.ui.t_dist_mat_path.text()):
            return
        if self.ui.threshold_edit.text() == '':
            return
        
        dataset = pickle.load(open(self.ui.t_dataset_path.text(), 'rb'))
        usage_matrix = dataset.get_usage_matrix()
        dist_mat = load_data(self.ui.t_dist_mat_path.text())
        test_file = self.ui.t_test_file_path.text()
        t = float(self.ui.threshold_edit.text())
        
        if self.ui.ii_radio_btn.isChecked():
            self.ui.status_label.setText('FC item/item...')
            w = Worker(fc_item_item, usage_matrix, dist_mat, test_file, threshold=t)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            w.signals.result.connect(self.set_t_results)
            self.tasks.start(w)
        else:
            self.ui.status_label.setText('FC user/user...')
            w = Worker(fc_user_user, usage_matrix, dist_mat, test_file, threshold=t)
            w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
            w.signals.result.connect(self.set_t_results)
            self.tasks.start(w)
    
    def set_t_results(self, res):
        self.ui.t_mae_label.setText("MAE = {:.5f}".format(res[0]))
        self.ui.t_rmse_label.setText("RMSE = {:.5f}".format(res[1]))

    #############################################################################

    def run_knn(self):
        if not os.path.isfile(self.ui.c_test_file_path.text()):
            return
        if not os.path.isfile(self.ui.c_dataset_path.text()):
            return
        if not os.path.isfile(self.ui.c_dist_mat_path_1.text()):
            return
        if self.ui.k_edit.text() == '':
            return
        
        dataset = pickle.load(open(self.ui.c_dataset_path.text(), 'rb'))
        usage_matrix = dataset.get_usage_matrix()
        dist_mat = load_data(self.ui.c_dist_mat_path_1.text())
        test_file = self.ui.c_test_file_path.text()
        k = int(self.ui.k_edit.text())
        knn = None
        if self.ui.multiview_check_box.isChecked():
            if self.ui.c_dist_mat_path_2.text() == "": return
            dist_mat_2 = load_data(self.ui.c_dist_mat_path_2.text())
            self.ui.status_label.setText('running KNN Multiview...')
            knn = KNNMultiview(k, dist_mat, dist_mat_2, usage_matrix)
        else:
            self.ui.status_label.setText('running KNN...')
            knn = KNN(k, dist_mat, usage_matrix)
        w = Worker(self.knn, knn, test_file)
        w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
        w.signals.result.connect(self.set_knn_results)
        self.tasks.start(w)

    def knn(self, knn_algo, test_file: str):
        test_dict = createDictTestMovies(test_file)
        knn_algo.process(test_dict)
        return knn_algo.mae_rmse()
    
    def set_knn_results(self, res):
        self.ui.knn_mae_label.setText("MAE = {:.5f}".format(res[0]))
        self.ui.knn_rmse_label.setText("RMSE = {:.5f}".format(res[1]))
    
    def run_kmedoids(self):
        if not os.path.isfile(self.ui.c_test_file_path.text()):
            return
        if not os.path.isfile(self.ui.c_dataset_path.text()):
            return
        if not os.path.isfile(self.ui.c_dist_mat_path_1.text()):
            return
        if self.ui.kmedoids_cluster_count_edit.text() == '':
            return
        
        dataset = pickle.load(open(self.ui.c_dataset_path.text(), 'rb'))
        usage_matrix = dataset.get_usage_matrix()
        dist_mat = load_data(self.ui.c_dist_mat_path_1.text())
        test_file = self.ui.c_test_file_path.text()
        kmeds = int(self.ui.kmedoids_cluster_count_edit.text())

        self.ui.status_label.setText('running K-medoids...')
        w = Worker(self.kmedoids, kmeds, usage_matrix, dist_mat, test_file)
        w.signals.finished.connect(lambda : self.ui.status_label.setText('Done'))
        w.signals.result.connect(self.set_kmedoids_results)
        self.tasks.start(w)

    def kmedoids(self, meds_count, usage_mat, dist_mat, test_file):
        test_dict = createDictTestMovies2(test_file)
        initial_medoids = np.random.choice(
            [i for i in range(len(usage_mat))],
            meds_count
        )
        #meds = [int(x) for x in initial_medoids]
        kmed = kmedoids(dist_mat,
            initial_medoids,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        (mae, rmse, prec, rec) = MAE_RMSE(usage_mat, dist_mat, kmed.get_clusters(), test_dict)
        if self.ui.bso_check_box.isChecked():
            bso = BSO(len(usage_mat), kmed.get_medoids(), lambda x: bso_eval(x, dist_mat, usage_mat, test_dict), from_count=False)
            (mae, rmse, prec, rec), meds = bso.run(int(self.ui.bso_flip_edit.text()))
        return (mae, rmse, prec, rec)

    def set_kmedoids_results(self, res):
        self.ui.kmedoids_mae_label.setText("MAE = {:.5f}".format(res[0]))
        self.ui.kmedoids_rmse_label.setText("RMSE = {:.5f}".format(res[1]))

class Worker(QRunnable):
    #sec_signal = pyqtSignal(str)
    def __init__(self, func, *args, **kwargs):
        super(Worker, self).__init__()
        self.exec = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        #self.go = True
    
    @pyqtSlot()
    def run(self):
        #this is a special fxn that's called with the start() fxn
        result = self.exec(*self.args, **self.kwargs)
        try:
            print('start worker')
            #result = self.exec(*self.args, **self.kwargs)
        except:
            print("worker error!")
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)

        