import numpy as np
from . import utils
from scipy import stats
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from pymatgen.core.periodic_table import Element

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
})

# LDS implemented GBT

class GBTRegressorPersistent():
    def __init__(self, allX, ally,
                 predict_column,
                 init_param,
                 test_size = 0.1,
                 toplot = True,
                 limlower = 0,
                 limupper = 100,
                 seed = 0,
                 keepfeatures = None,
                 holdlower = True,
                 holdupper = True,
                 param_grid = None,
                 additional_holdout = {},
                 **wparams
                ):
        self.feature_names = allX.columns
        self.scaler = StandardScaler()
        self._allX = self.scaler.fit_transform(allX.values, ally.values)
        self._ally = ally
        self.train_idx = None
        self.test_idx = None
        self._allweights = self.prepare_weights(**wparams)
        self._predict_column = predict_column
        self._hparams = init_param
        self._test_size = test_size
        self._toplot = toplot
        self._limlower = limlower
        self._limupper = limupper
        self._seed = seed
        self._keepfeatures = keepfeatures
        self._holdlower = holdlower
        self._holdupper = holdupper
        self._param_grid = param_grid
        self._additional_holdout = additional_holdout

    @property
    def hparams(self):
        """The current hyperparameters used for ML model training, before calling the `hyp_param_search` function, it will be the one you passed in."""
        return self._hparams

    @hparams.setter
    def hparams(self, hyper_params):
        print("Hyperparameters set")
        self._hparams = hyper_params
        
    @property
    def param_grid(self):
        """The current hyperparameters used for ML model training, before calling the `hyp_param_search` function, it will be the one you passed in."""
        return self._param_grid

    @param_grid.setter
    def param_grid(self, new_grid):
        print("Grid of Hyperparameters for tuning set")
        self._param_grid = new_grid    
                  
    def _focus(self, mu, sigmag, alpha = None, amp = None):
        """Prepare focus function from parameters for focussed LDS"""
        try:
            all_labels = self._ally.values.reshape(-1).astype(float)
        except:
            all_labels = self._ally.reshape(-1).astype(float)
        
        if amp is None:
            amp = 1
        if alpha is None:
            focus_func = amp * np.exp(-(all_labels - mu) ** 2  / (2 * sigmag ** 2)).reshape(-1)
        else:
            normpdf = (1 / (sigmag * np.sqrt(2 * np.pi))) * np.exp(-(np.power((all_labels - mu), 2) / (2 * np.power(sigmag, 2))))
            normcdf = (0.5 * (1 + erf((alpha * ((all_labels - mu) / sigmag)) / (np.sqrt(2)))))
            focus_func = (amp *  normpdf * normcdf).reshape(-1)
        return focus_func
    
    def prepare_weights(self, reweight = 'sqrt_inv', min_target = None, max_target = None, nbins = 60, lds = True,
                    lds_kernel = 'gaussian', lds_ks = 2, lds_sigma = 3, focus = None):
        """Prepare weights for labels to enable label distribution smoothing (LDS)"""
        assert reweight in {'none', 'inverse', 'sqrt_inv', 'log_inv', 'exp_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to  \'log_inv\' (default), \'exp_inv\', \'sqrt_inv\'  or \'inverse\' when using LDS"
        
        value_range = self._ally.max() - self._ally.min()
    
        if min_target is None:
            min_target = round(self._ally.min() - value_range/nbins, 2)

        if max_target is None:
            max_target = round(self._ally.max() + value_range/nbins, 2)
            
        bins = np.linspace(min_target, max_target, nbins)
        
        try:
            all_labels = self._ally.values.reshape(-1)
        except:
            all_labels = self._ally.reshape(-1)
        
        label_locs = np.abs(all_labels.reshape(-1, 1) - bins).argmin(axis = 1)
        
        counts = np.zeros(len(bins))
        for i in range(len(bins)):
            counts[i] = (label_locs == i).sum()
        
        lds_kernel_window =  utils.get_lds_kernel_window(lds_kernel,
                                                             lds_ks,
                                                             lds_sigma)
        smoothed_counts = convolve1d(counts, weights = lds_kernel_window, mode = 'constant')
        
        if reweight == 'sqrt_inv':
            smoothed_counts = np.sqrt(np.clip(smoothed_counts, 1, counts.max()))
        
        elif reweight == 'inverse':
            smoothed_counts = np.clip(smoothed_counts, 1, smoothed_counts.max())

        elif reweight == 'log_inv':
            smoothed_counts = np.log(smoothed_counts + np.e)

        elif reweight == 'exp_inv':
            smoothed_counts = np.exp(smoothed_counts/20)
            
        num_per_label = np.asarray([smoothed_counts[x] for x in label_locs])
        
        if not len(num_per_label) or reweight == 'none':
            return None
        
        print(f"Using with re-weighting: [{reweight.upper()}]")
        
        if lds:
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            if reweight == 'exp_inv':
                weights = 0.05 * 0.95/num_per_label
            else:
                weights = 1 / num_per_label
            
            if focus is not None:
                focus_func = self._focus(**focus)
                weights = weights + focus_func
                weights = weights/weights.max()

            scaling = len(weights) / weights.sum()
            weights = scaling * weights
            return weights.reshape(-1)
        
    def run(self, retrain = True):
        print('Model hyperparameters:')
        print("-"*53)
        for key, value in self._hparams.items():
            print('{}: {}'.format(key, value))
        print("="*53)
        
        keep_indices, holdout_indices =\
            utils.filter_by_predict_value(self._limlower, self._limupper,
                                    self._ally, self._holdlower, self._holdupper)

        # set up training data
        self._X = np.asarray(self._allX)[keep_indices]
        self._y = np.asarray(self._ally)[keep_indices]
        if self._allweights is not None:
            self._weights = np.asarray(self._allweights)[keep_indices]

        # setup kept out data if outside limits
        self._Xhold = np.asarray(self._allX)[holdout_indices]
        self._yhold = np.asarray(self._ally)[holdout_indices]
        if self._allweights is not None:
            self._weightshold = np.asarray(self._allweights)[holdout_indices]

        # test/train split setup
        self._nsplits = int(np.ceil(1/self._test_size))
        self._kf = KFold(n_splits = self._nsplits, shuffle = True, random_state = self._seed)
        self._kf.get_n_splits(self._X)
        self._modelstats = []

        # store results of each kfold
        self._all_train_pred = []
        self._all_train_mae = []
        self._all_test_pred = []
        self._all_test_mae = []
        self._all_hold_pred = []
        self._all_hold_mae = []
        self._all_feature_importance = np.zeros((self._nsplits, np.shape(self._X)[1]))

        if retrain:
            self._all_est = []
        
        print(f"Training for {self._predict_column}")
        print("-"*53)
        print('X shape: ', np.shape(self._X))
        print('y shape: ', np.shape(self._y))
        print("K-fold | Train MAE | Test MAE | Max(y) | Min(y)")
        print("-"*53)
        
        for it, (train_index, test_index) in enumerate(self._kf.split(self._X)):
            try:
                X_train, y_train, weights_train = self._X[train_index], self._y[train_index], self._weights[train_index]
                X_test, y_test, weights_test = self._X[test_index], self._y[test_index], self._weights[test_index]
                
            except:
                X_train, y_train = self._X[train_index], self._y[train_index]
                X_test, y_test = self._X[test_index], self._y[test_index]
                
            if retrain:
                if self._allweights is not None:
                    est = GradientBoostingRegressor(**self._hparams).fit(X_train, y_train, sample_weight = weights_train)
                else:
                    est = GradientBoostingRegressor(**self._hparams).fit(X_train, y_train, sample_weight = None)
                self._all_est.append(est)
            else:
                est = self._all_est[it]

            # evaluate model on the training set
            y_train_pred = est.predict(X_train)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            self._all_train_pred.append((y_train, y_train_pred))
            self._all_train_mae.append(train_mae)

            # evaluate model on the test set
            y_test_pred = est.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            self._all_test_pred.append((y_test,y_test_pred))
            self._all_test_mae.append(test_mae)

            # evaluate model on the holdout set
            if len(self._yhold) != 0:
                yhold_pred = est.predict(self._Xhold)
                holdout_mae = mean_absolute_error(self._yhold, yhold_pred)
                self._all_hold_pred.append((self._yhold, yhold_pred))
                self._all_hold_mae.append(holdout_mae)

            # evaluate model on any additional holdout sets
            for key in self._additional_holdout.keys():
                thisy = self._additional_holdout[key]['yhold']
                thisy_pred = est.predict(self._additional_holdout[key]['Xhold'])
                this_mae = mean_absolute_error(thisy, thisy_pred)
                print(this_mae)
                self._additional_holdout[key]['allhold_pred'].append((thisy, thisy_pred))
                self._additional_holdout[key]['allhold_mae'].append(this_mae)

            print("%d\t%.2f\t%.2f\t%.2f\t%.2f"%(it, train_mae, test_mae, np.max(y_test), np.min(y_test)))
            self._modelstats.append([self._seed, train_mae, test_mae])
            
            # Feature importance
            feature_importance = est.feature_importances_
            # normalize relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            self._all_feature_importance[it,:] = feature_importance
        
        try:
            self._finalest = GradientBoostingRegressor(**self._hparams).fit(self._X, self._y, sample_weight = self._weights)
        except:
            self._finalest = GradientBoostingRegressor(**self._hparams).fit(self._X, self._y)
        if self._toplot:
            self.plot_training()

    def hparams_search(self, opt_train = True):
        keep_indices, holdout_indices =\
        utils.filter_by_predict_value(self._limlower, self._limupper,
                                self._ally, self._holdlower, self._holdupper)
        # set up training data
        X_train = np.array(self._allX)[keep_indices]
        y_train = np.array(self._ally)[keep_indices]
        weights_train = np.array(self._allweights)[keep_indices]
        
        est = GradientBoostingRegressor()
        
        if self._param_grid is None:
            self._param_grid = {'learning_rate': [0.01, 0.005, 0.002, 0.001, 0.0005],
                          'max_depth': [4, 5, 6, 7, 8],
                          'min_samples_leaf': [3, 5, 9, 17],
                         }
            
        gs_cv = GridSearchCV(
            est,
            self._param_grid,
            cv = 10, # k-fold
            n_jobs = 4, # 
            verbose = 1
        )
        
        gs_cv.fit(X_train, y_train, weights_train)
        
        self.hparams.update(gs_cv.best_params_)
        
        if opt_train:
            self.run()
 
    
    def plot_training(self):

        ncols=4
        fig, ax = plt.subplots(nrows = self._nsplits, ncols = ncols,
                               figsize=(3.3*ncols,1.9*self._nsplits),
                               gridspec_kw={'width_ratios': [3,3,3,1]},
                               constrained_layout=True)
        
        for it, (train_index, test_index) in enumerate(self._kf.split(self._X)):

            X_train, X_test = self._X[train_index], self._X[test_index]
            y_train, y_test = self._y[train_index], self._y[test_index]
            
            ######################################################################
            # Train/test parity plot for each k-fold
            ######################################################################
            # Training parity plot
            ax[it,0].scatter(self._all_train_pred[it][0],self._all_train_pred[it][1],
                             edgecolor='blue', linewidths=1, alpha=0.3,
                             label="Train MAE = %.2f"%(self._all_train_mae[it]))
            ax[it,0].set_xlabel(r"True %s"%column_to_label(self._predict_column))
            ax[it,0].set_ylabel(r"Model %s"%column_to_label(self._predict_column))
            draw_y_equals_x(ax[it,0])
            ax[it,0].legend(loc='best')

            # Test parity plot
            SC = stats.spearmanr(self._all_test_pred[it][0],self._all_test_pred[it][1])
            ax[it,1].scatter(self._all_test_pred[it][0],self._all_test_pred[it][1],
                             edgecolor='blue', linewidths=1, alpha=0.3,
                             label="Val MAE (SC)= %.2f (%.2f)"%(self._all_test_mae[it],SC[0]))

            # Plot holdout with special color
            if len(self._yhold) != 0:
                ax[it,1].scatter(self._all_hold_pred[it][0],self._all_hold_pred[it][1],
                                 edgecolor='red', linewidths=1, alpha=0.3,
                                 labeall="Holdout MAE = %.2f"%(self._all_holdout_mae[it]))
            ax[it,1].set_xlabel(r"True %s"%column_to_label(self._predict_column))
            ax[it,1].set_ylabel(r"Model %s"%column_to_label(self._predict_column))
            draw_y_equals_x(ax[it,1])
            ax[it,1].legend(loc='best')


            # Learning curve plot
            test_score = np.zeros((self._hparams['n_estimators'],), dtype=np.float64)
            for i, y_pred in enumerate(self._all_est[it].staged_predict(X_test)):
                test_score[i] = self._all_est[it].loss_(self._all_test_pred[it][0], y_pred)

            # Plot staged predicitions
            ax[it,2].plot(np.arange(self._hparams['n_estimators']) + 1, 
                          self._all_est[it].train_score_, 'b-', label='Training Set Deviance')
            ax[it,2].plot(np.arange(self._hparams['n_estimators']) + 1, 
                          test_score, 'r-', label='Val Set Deviance')
            ax[it,2].legend(loc='upper right')
            ax[it,2].set_xlabel('Boosting Iterations')
            ax[it,2].set_ylabel('Deviance')


            # Feature importance plot
            feature_importance = self._all_feature_importance[it]
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            maxdisplay = min(len(pos),8) # we only want to plot a max num of features

            ax[it,3].barh(pos[-maxdisplay:], feature_importance[sorted_idx][-maxdisplay:], align='center')
            ax[it,3].set_yticks(pos[-maxdisplay:])
            if type(self._allX) == np.ndarray:
                pass
            else:
                ticklabels = [feature.replace('_','\_')\
                              for feature in self._allX.columns[sorted_idx][-maxdisplay:]]
                ax[it,3].set_yticklabels(ticklabels)
            ax[it,3].set_xlabel('Relative Importance')
            ax[it,3].set_xlim((0, 100))

        plt.show()
        plt.close()