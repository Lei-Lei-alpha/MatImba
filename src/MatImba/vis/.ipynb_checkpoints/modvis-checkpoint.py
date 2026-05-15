import os
# import shape
import torch
import traceback
import numpy as np
from ..utils import utils
from ..utils import data
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
})

def draw_y_equals_x(ax, colour = '#d73027', **kwargs):
    colour = kwargs.pop('c', colour)
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    ax.plot(
        [max(xlim[0],ylim[0]), min(xlim[1],ylim[1])],
        [max(xlim[0],ylim[0]), min(xlim[1],ylim[1])],
        c = colour, **kwargs
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def vertical_img_concat(flist, savename):
    images = [Image.open(x) for x in flist]
    widths, heights = zip(*(i.size for i in images))
    print(widths,heights)
    
    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    if savename is not None:
        new_im.save(savename)

def horizontal_img_concat(flist, savename):
    
    images = [Image.open(x) for x in flist]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    if savename is not None:
        new_im.save(savename)
        
    
def plot_special_marker_fcn(ax, thiscomplist, setcombinations, xvals, yvals, edgecolor = "#4d4d4d",
            marker="*",s = 100, colour = "#b2182b",linewidths=0.5):
    """
    To plot a specific composition as its own marker
    """

    thiscompset = set(thiscomplist)
    thiscompstr = "".join(sorted(thiscomplist, key=lambda x: Element(x).Z))
    #thiscompstr = r"$\mathrm{"+thiscompstr+"}$"
    thiscompind = setcombinations.index(thiscompset)
    ax.scatter(xvals[thiscompind],yvals[thiscompind],edgecolor=edgecolor,
               marker=marker,s=s, c=c,label=thiscompstr,linewidths=linewidths)

    print("%s , x=%.2f , y=%.2f"%(thiscomplist,
                                              xvals[thiscompind],
                                              yvals[thiscompind]))
def column_to_label(colname):
    name_map = {
        'Heat_of_Formation_kJperMolH2': r"$\Delta H$ [kJ/mol H$_2$]",
        'Entropy_of_Formation_JperMolH2perK': r"$\Delta S$ [J/(mol H$_2$ $\cdot$ K)]",
        'Equilibrium_Pressure_25C': r"$P_{eq}^o$",
        'volume': r"$V_{cell}$ [\AA$^3$]",
        'volume_ps': r"$\nu_{pa}^{MP}$ [\AA$^3$/atom]",
        'volume_ps_generic': r"$\bar{\nu_{pa}}$ [\AA$^3$/atom]",
        'mean_GSvolume_pa': r"$\bar{\nu}_{\mathrm{pa}}^{\mathrm{Magpie}}$ [\AA$^3$/atom]",
        'empty_volume_ps': r"$\nu_{pa}-\bar{V}_{atom}$ [\AA$^3$/atom]",
        'mean_CovalentRadius': r"mean$\_$CovalentRadius",
        'mean_SpaceGroupNumber': r"mean$\_$SpaceGroupNumber",
        'energy_per_atom': r"$E_{atom}$",
        'formation_energy_per_atom': r"$E_{f,atom}$",
        'mean_Electronegativity': r"mean$\_$Electronegativity",
        'most_Electronegativity': r"most$\_$Electronegativity",
        'mean_MeltingT': r"mean$\_$MeltingT",
        'normalized_delH': r"$\Delta H / (RT^o)$",
        'normalized_delS': r"$\Delta S / R$",
        'Hydrogen_Weight_Percent': r"$\mathrm{H }wt.\%$",
        'HtoM': r"$\mathrm{H/M}$",
        'd(lnP_D)/d(H/M)': r"${d(\ln P _\mathrm{D})} / {d \mathrm{(H/M)}}$",
        'Ef_eV': r"$\mathrm{E}_{F}\mathrm{ [eV]}$",
    }
    
    if colname in name_map:
        return name_map[colname]
    else:
        print("Add translation for %s"%colname)
        return colname 
    
def prettify_magpie(s):
    if s not in data.magpie_features:
        return s
    else:
        return s if data.magpie_features[s] is None else data.magpie_features[s]


def get_shap_values(model, features, top = 8):
    if hasattr(model,'_finalest'):
        m = model._finalest
    else:
        m = model
    fea = deepcopy(features)
    fea.columns = [prettify_magpie(col) for col in fea.columns]
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(fea)
    shap_obj = explainer(fea)
    return shap_values

def summarize_shap_values(shap_values, features, savename = 'tmp', figsize = None, top = 6, plot_type = "dot", cmap = 'coolwarm'):
    """
    plot_type: layered_violin, violin or dot 
    """
    
    print(type(features))
    fea = deepcopy(features)

    try:
        #plt.rcParams['figure.constrained_layout.use'] = True
        print(fea.columns)
        fea.columns = [prettify_magpie(col) for col in fea.columns]
        # formula for default plot size = (1.5 * max_display + 1, 0.8 * max_display + 1)
        if figsize is not None:
            shap.summary_plot(shap_values, fea, max_display = top, show=False,
                              plot_size = figsize, plot_type = plot_type, color = cmap)
        else:
            shap.summary_plot(shap_values, fea, max_display = top,
                              show=False, plot_type = plot_type, color = cmap)
        #shap.plots.beeswarm(shap_obj, max_display=top, plot_size=(3.0,2.2))

        plt.gcf().axes[-1].set_aspect(26)
        plt.gcf().axes[-1].set_box_aspect(26)
        # print(shap_values)
        
    except Exception:
        fea = deepcopy(features)
        traceback.print_exc()   
        
def plot_summary(GBTobj, fea, titles = None, savename = None, prettify_Magpie = True,
                 limlower = None, limupper = None, HEIGHT = 3, DPI = 600, savetemp = False):

    ######################################################################
    # Plot publication figure of combined k-fold train and test
    ######################################################################
    
    # so we can recalculate MAE over target range
    all_y_train_true = []
    all_y_train_pred = []
    all_y_test_true = []
    all_y_test_pred = []

    # loop over all kfolds
    numKfolds = len(GBTobj._all_test_pred)
    alltestMAEs = []
    
    for i in range(numKfolds):
        
        # Filter values if in the range over which we actually care about 
        # the error metric
        y_train_true = GBTobj._all_train_pred[i][0]
        y_train_pred = GBTobj._all_train_pred[i][1]
        keep_indices, _ = utils.filter_by_predict_value(limlower, limupper, y_train_true)
        y_train_true = y_train_true[keep_indices]
        y_train_pred = y_train_pred[keep_indices]
        
        y_test_true = GBTobj._all_test_pred[i][0]
        y_test_pred = GBTobj._all_test_pred[i][1]
        keep_indices, _ = utils.filter_by_predict_value(limlower, limupper, y_test_true)
        y_test_true = y_test_true[keep_indices]
        y_test_pred = y_test_pred[keep_indices]
        
        all_y_train_true += list(y_train_true)
        all_y_train_pred += list(y_train_pred)
        
        all_y_test_true += list(y_test_true)
        all_y_test_pred += list(y_test_pred)
        
        testMAE = np.average(np.abs(np.array(all_y_test_true)-np.array(all_y_test_pred)))
        alltestMAEs.append(testMAE)

        trainMAE = np.average(np.abs(np.array(all_y_train_true)-np.array(all_y_train_pred))) 

    ######################################################################
    #K-fold concatenated train and test sets
    ######################################################################

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(6.5, HEIGHT), sharey = True, constrained_layout=True)

    trainlabel = r"$\langle$MAE$\rangle_{\mathrm{Train}}$ = %.2f"%(trainMAE)#np.average(GBTobj._all_train_mae))
    testlabel = r"$\langle$MAE$\rangle_{\mathrm{Test}}$ = %.2f"%(testMAE)#np.average(GBTobj._all_test_mae))

    ax[0].scatter(all_y_train_true, all_y_train_pred, edgecolor = None,
                                                        color = '#2166ac',
                                                        linewidths = 0,
                                                        alpha = 0.1)
    ax[1].scatter(all_y_test_true, all_y_test_pred, edgecolor = None,
                                                        color = '#2166ac',
                                                        linewidths = 0,
                                                        alpha = 0.1)        
        
    draw_y_equals_x(ax[0])
    draw_y_equals_x(ax[1])
    
    ax[0].text(0.5, 0.9, trainlabel, va = 'center', ha = 'center', transform = ax[0].transAxes)
    ax[1].text(0.5, 0.9, testlabel, va = 'center', ha = 'center', transform = ax[1].transAxes)
    ax[0].set_ylabel(r"Predicted %s"%column_to_label(GBTobj._predict_column))
    ax[0].set_xlabel(r"Experimental %s"%column_to_label(GBTobj._predict_column))
    ax[1].set_xlabel(r"Experimental %s"%column_to_label(GBTobj._predict_column))
    
    plt.savefig('/var/tmp/figTrainTest.png', dpi = DPI)
    plt.savefig('/var/tmp/figTrainTest.pdf')
    plt.show()
    plt.close()
    
    ######################################################################
    #K-fold concatenated test sets
    ######################################################################
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (3.6, HEIGHT), sharey = True, constrained_layout=True)

    testlabel = r"$\langle \mathrm{MAE} \rangle_K$ = %.2f"%(testMAE)#np.average(GBTobj._all_test_mae))

    cm = plt.cm.get_cmap('terrain_r')
    
    sc = ax.hist2d(all_y_test_true, all_y_test_pred, bins = 50, vmin = 0, vmax = 7,
               cmap = cm, label = testlabel)

    cb = plt.colorbar(sc[3], ax = [ax])

    cb.ax.set_title(r'$ρ _{\mathrm{S}}$') # density of training examples  
        
    draw_y_equals_x(ax)
    
    print(np.average(alltestMAEs), np.std(alltestMAEs))
    
    ax.set_ylabel(r"Predicted %s"%column_to_label(GBTobj._predict_column))
    ax.set_xlabel(r"Experimental %s"%column_to_label(GBTobj._predict_column))
    
    if titles is not None:
        plt.title(titles[1])
    plt.savefig('/var/tmp/figTest.pdf')
    plt.savefig('/var/tmp/figTest.png',dpi=DPI)
    plt.show()
    plt.close()
    
    
    ######################################################################
    # SHAP analysis
    ######################################################################
    # dH models
    #plt.rcParams['figure.constrained_layout.use'] = True
    shapvals = get_shap_values(GBTobj, fea)
    summarize_shap_values(shapvals, fea, figsize=(3.8, HEIGHT), top = 5)
    plt.xlabel('SHAP value')
    if titles is not None:
        plt.title(titles[2])
    plt.tight_layout(pad = 0.15)
    plt.savefig('/var/tmp/figSHAP.pdf')
    plt.savefig('/var/tmp/figSHAP.png', dpi = DPI)
    plt.show()
    plt.close()

    ######################################################################
    # Plot publication figure of average importances over all k-fold models
    ######################################################################
    fig, ax = plt.subplots(figsize=(3.6, HEIGHT), constrained_layout = True)

    all_feature_importance = np.sum(GBTobj._all_feature_importance, axis=0) / (GBTobj._nsplits)
    sorted_idx = np.argsort(all_feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5
    maxdisplay = min(len(pos), 5) # we only want to plot a max num of features

    ax.barh(pos[-maxdisplay:], all_feature_importance[sorted_idx][-maxdisplay:],
            align = 'center')
    ax.set_yticks(pos[-maxdisplay:])
    
    if prettify_Magpie:
        ticklabels = [prettify_magpie(features)\
                      for features in GBTobj.feature_names[sorted_idx][-maxdisplay:]]

    else:
        ticklabels = [features.replace('_','\_')\
                      for features in GBTobj.feature_names[sorted_idx][-maxdisplay:]]

    ax.set_yticklabels(ticklabels)
    
    ax.set_xlabel(r'$\langle$Relative Importance$\rangle$')
    ax.set_xlim((0, 100))

    if titles is not None:
        plt.title(titles[3])
    plt.savefig('/var/tmp/figFeature.png',dpi = DPI)

    ######################################################################
    # Plot the MAE of subsets of the data binned on the true predict value
    ######################################################################
    nbins = 15
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.set_position([0.18, 0.18, 0.65, 0.75])
    axtwin = ax.twinx()
    axtwin.set_position([0.18, 0.18, 0.65, 0.75])

    ys, preds = np.hstack(GBTobj._all_test_pred)
    if limlower != None and limupper != None:
        keep_idx = np.where((ys >= limlower) & (ys <= limupper))
        ys, preds = ys[keep_idx], preds[keep_idx]
    maes = np.abs(ys - preds)
    
    hist, bin_edges = np.histogram(ys, bins = nbins)
    x = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = (bin_edges[-1] - bin_edges[0])/nbins
    ys, bin_edges = ys.astype(float), bin_edges.astype(float)
    label_locs = np.fmin(np.digitize(ys, bin_edges), nbins)
    binned_AEs = np.zeros(len(x))
    
    for i in range(len(x)):
        locs = np.where(label_locs == i + 1)
        binned_AEs[i] = maes[locs].mean()

    x = x[~np.isnan(binned_AEs)]
    binned_AEs = binned_AEs[~np.isnan(binned_AEs)]
    
    ax.hist(ys, color = '#2166ac', bins = nbins, width = 0.85 * bin_width, edgecolor="#878787", linewidth= .5,)
    ax.set_xlabel(r"Experimental %s"%column_to_label(GBTobj._predict_column))
    ax.set_ylabel(r"HEA4HST Frequency")
    ax.tick_params(axis='y', colors='#2166ac')
    ax.yaxis.label.set_color('#2166ac')

    axtwin.plot(x, binned_AEs, c = "#b2182b", marker="s", markerfacecolor="#fddbc7")
    axtwin.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ within bin")
    axtwin.tick_params(axis='y', colors='#b2182b')
    axtwin.yaxis.label.set_color('#b2182b')

   
    if titles is not None:
        plt.title(titles[3])
    plt.savefig('/var/tmp/figMAEdist.pdf')
    plt.savefig('/var/tmp/figMAEdist.png', dpi=DPI)
    
    sep = np.ones(int(DPI*HEIGHT)*10, dtype=np.uint8)
    sep = np.reshape(sep,(int(DPI*HEIGHT),1*10))
    im = Image.fromarray(sep)
    im.save("/var/tmp/sep.png")

    # too annoying to get the subplot sizing correct in matplotlib, 
    # so just save the images and recombine with
    flist = ['/var/tmp/figTest.png', '/var/tmp/sep.png', '/var/tmp/figSHAP.png', '/var/tmp/sep.png', '/var/tmp/figMAEdist.png']
    
    horizontal_img_concat(flist, savename)
    
    if savetemp:
        [flist.remove(x) for x in flist if x == '/var/tmp/sep.png']
        flist.append('/var/tmp/figTrainTest.png')
        
        names = [file.rsplit('/',1)[-1] for file in flist]
        fig_dir = savename.rsplit('/', 1)[0]
        names = [os.path.join(fig_dir, name) for name in names]
        [os.system(f'cp {file} {name}') for file, name in zip(flist, names)]

    plt.show()

def mae_within_bin(gbr_model, nbins = 15, limlower = None, limupper = None, filename = None):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.set_position([0.18, 0.18, 0.65, 0.75])

    ys, preds = np.hstack(gbr_model._all_test_pred)
    if limlower != None and limupper != None:
        keep_idx = np.where((ys >= limlower) & (ys <= limupper))
        ys, preds = ys[keep_idx], preds[keep_idx]
    maes = np.abs(ys - preds)
    
    hist, bin_edges = np.histogram(ys, bins = nbins)
    x = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = (bin_edges[-1] - bin_edges[0])/nbins
    label_locs = np.fmin(np.digitize(ys, bin_edges), nbins)
    binned_AEs = np.zeros(len(x))
    
    for i in range(len(x)):
        locs = np.where(label_locs == i + 1)
        binned_AEs[i] = maes[locs].mean()

    x = x[~np.isnan(binned_AEs)]
    binned_AEs = binned_AEs[~np.isnan(binned_AEs)]
    
    ax.hist(ys, color = '#2166ac', bins = nbins, width = 0.85 * bin_width, edgecolor = "#878787", linewidth= .5,)
    ax.set_xlabel(r"Experimental %s"%column_to_label(gbr_model._predict_column))
    ax.set_ylabel(r"HEA4HST Frequency")
    ax.tick_params(axis='y', colors='#2166ac')
    ax.yaxis.label.set_color('#2166ac')
    
    axtwin = ax.twinx()
    axtwin.plot(x, binned_AEs, c = "#b2182b", marker="s", markerfacecolor="#fddbc7")
    axtwin.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ within bin")
    axtwin.tick_params(axis='y', colors='#b2182b')
    axtwin.yaxis.label.set_color('#b2182b')
    
    axtwin.set_position([0.18, 0.18, 0.65, 0.75])
       
    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()

def compare_mwb(gbr_models_lst, split = 'test', nbins = 15, labels = None,
                limlower = None, limupper = None, filename = None):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.set_position([0.18, 0.18, 0.65, 0.75])
    axtwin = ax.twinx()
    axtwin.set_position([0.18, 0.18, 0.65, 0.75])

    markers = ['s', 'o', '^', 'v', '<', '>', 'd', 'p', '*', 'h', 'H', '8', 'P', 'X']
    colours = list(mcolors.TABLEAU_COLORS.values())

    if labels is None:
        labels = [gbr_model._name for gbr_model in gbr_models_lst]
       
    for i, gbr_model in enumerate(gbr_models_lst):
        if split == 'test':
            ys, preds = np.hstack(gbr_model._all_test_pred)
        elif split == "train":
            ys, preds = np.hstack(gbr_model._all_train_pred)
        if limlower != None and limupper != None:
            keep_idx = np.where((ys >= limlower) & (ys <= limupper))
            ys, preds = ys[keep_idx], preds[keep_idx]
        maes = np.abs(ys - preds)
        hist, bin_edges = np.histogram(ys, bins = nbins)
        x = (bin_edges[:-1] + bin_edges[1:])/2
        bin_edges = bin_edges.astype(float)
        y_locs = np.fmin(np.digitize(ys, bin_edges), nbins)
        binned_AEs = np.zeros(len(x))
        
        for j in range(len(x)):
            locs = np.where(y_locs == j + 1)
            binned_AEs[j] = maes[locs].mean()

        x = x[~np.isnan(binned_AEs)]
        binned_AEs = binned_AEs[~np.isnan(binned_AEs)]
        
        ax.plot(x, binned_AEs, c = colours[i], marker = markers[i], markerfacecolor = "#f0f0f0",
                label = labels[i])

    bin_width = (bin_edges[-1] - bin_edges[0])/nbins
    axtwin.hist(ys, color = '#2166ac', bins = nbins, width=0.85 * bin_width,
                edgecolor="#878787", linewidth= .5)
    axtwin.set_ylabel(r"HEA4HST Frequency")
    axtwin.tick_params(axis='y', colors='#2166ac')
    axtwin.yaxis.label.set_color('#2166ac')
    
    
    ax.set_zorder(axtwin.get_zorder()+1)
    ax.patch.set_visible(False)
    
    ax.set_xlabel(r"Experimental %s"%column_to_label(gbr_models_lst[0]._predict_column))
    if split == 'test':
        ax.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ within bin")
    elif split == "train":
        ax.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Train}}$ within bin")
    ax.tick_params(axis='y')
    # ax.yaxis.label.set_color('#b2182b')

    ax.legend(loc = 0,framealpha = 0.2)
    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()
     
def plot_maes(model, train_loader, val_loader, target_name, filename = None):
    train_preds, train_targets = model.predict(train_loader)
    val_preds, val_targets = model.predict(val_loader)
    
    train_mae = mean_absolute_error(train_targets.numpy(), train_preds.numpy())
    val_mae = mean_absolute_error(val_targets.numpy(), val_preds.numpy())
    
    label_min = torch.concat([train_targets, val_targets]).flatten().min()
    label_max = torch.concat([train_targets, val_targets]).flatten().max()
    
    a = np.linspace(label_min, label_max, 20)
    b = a
    
    fig, axes = plt.subplots(1, 2, figsize = (6.5, 2.8), constrained_layout = True)
    axes[0].scatter(train_preds, train_targets, c = 'tab:blue', s = 40, linewidths = 0, alpha = 0.6)
    axes[0].plot(a, b, color='tab:red', lw = 2, alpha = 0.6)
    axes[0].text(0.5, 0.9, 'Train MAE = '+ str(round(train_mae, 2)),
                 va = 'center', ha = 'center', transform = axes[0].transAxes)

    axes[1].scatter(val_preds, val_targets, c = 'tab:blue', s = 40, linewidths = 0, alpha = 0.6)
    axes[1].plot(a, b, color='tab:red', lw = 2, alpha = 0.6)
    axes[1].text(0.5, 0.9, 'Test MAE = '+ str(round(val_mae, 2)),
                 va = 'center', ha = 'center', transform = axes[1].transAxes)
    
    [ax.set_xlabel(f'Predicted {target_name}') for ax in axes]
    [ax.set_ylabel(f'Experimental {target_name}') for ax in axes]
    
    if filename is not None:
        plt.savefig(filename, dpi=600)
    plt.show()
    
    
def binned_mae(model, data_loader, nbins = 15, limlower = None, limupper = None, filename = None, return_data = True):
    
    """For ResNet models"""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.set_position([0.18, 0.18, 0.65, 0.75])

    preds, ys = model.predict(data_loader)
    preds, ys = preds.numpy(), ys.numpy()
    
    if limlower != None and limupper != None:
        keep_idx = np.where((ys >= limlower) & (ys <= limupper))
        ys, preds = ys[keep_idx], preds[keep_idx]
    maes = np.abs(ys - preds)
    
    hist, bin_edges = np.histogram(ys, bins = nbins)
    x = (bin_edges[:-1] + bin_edges[1:])/2
    bin_width = (bin_edges[-1] - bin_edges[0])/nbins
    label_locs = np.fmin(np.digitize(ys, bin_edges), nbins)
    binned_AEs = np.zeros(len(x))
    
    for i in range(len(x)):
        locs = np.where(label_locs == i + 1)
        binned_AEs[i] = maes[locs].mean()
    # max_mae = binned_AEs.max()
    
    ax.hist(ys, color = '#2166ac', bins = nbins, width = 0.85 * bin_width, edgecolor = "#878787", linewidth= .5,)
    ax.set_xlabel(r"Experimental $\Delta$H")
    ax.set_ylabel(r"HEA4HST Counts")
    ax.tick_params(axis='y', colors='#2166ac')
    ax.yaxis.label.set_color('#2166ac')
    
    x1 = x[~np.isnan(binned_AEs)]
    hist = hist[~np.isnan(binned_AEs)]
    binned_AEs = binned_AEs[~np.isnan(binned_AEs)]
    pearson_cor = np.corrcoef(hist, y = binned_AEs)
    
    axtwin = ax.twinx()
    axtwin.plot(x1, binned_AEs, c = "#b2182b", marker="s", markerfacecolor="#fddbc7")
    axtwin.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ within bin")
    axtwin.tick_params(axis='y', colors='#b2182b')
    axtwin.yaxis.label.set_color('#b2182b')
    # ymin, ymax = axtwin.get_ylim()
    # axtwin.set_ylim(ymin, min(ymax, 20))
    
    axtwin.set_position([0.18, 0.18, 0.65, 0.75])
       
    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()
    
    if return_data:
        return x1, hist, binned_AEs, pearson_cor
    
    
def compare_nn_mwb(models_lst, data_loader_lst, labels = None, split = 'test', nbins = 15,
                limlower = None, limupper = None, bin_maes = {}, filename = None):
    """Compare MAE within bin of NN models"""
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.set_position([0.12, 0.18, 0.65, 0.75])
    axtwin = ax.twinx()
    axtwin.set_position([0.18, 0.18, 0.65, 0.75])

    markers = ['s', 'o', '^', 'v', '<', '>', 'd', 'p', '*', 'h', 'H', '8', 'P', 'X']
    colours = list(mcolors.TABLEAU_COLORS.values())

    if labels is None:
        labels = ['']*len(models_lst)
       
    for i, model, data_loader in zip(range(6), models_lst, data_loader_lst):
        
        preds, ys = model.predict(data_loader)
        preds, ys = preds.numpy(), ys.numpy()
    
        if limlower != None and limupper != None:
            keep_idx = np.where((ys >= limlower) & (ys <= limupper))
            ys, preds = ys[keep_idx], preds[keep_idx]
        maes = np.abs(ys - preds)
        hist, bin_edges = np.histogram(ys, bins = nbins)
        x = (bin_edges[:-1] + bin_edges[1:])/2
        bin_edges = bin_edges.astype(float)
        y_locs = np.fmin(np.digitize(ys, bin_edges), nbins)
        binned_AEs = np.zeros(len(x))
        
        for j in range(len(x)):
            locs = np.where(y_locs == j + 1)
            binned_AEs[j] = maes[locs].mean()

        x = x[~np.isnan(binned_AEs)]
        binned_AEs = binned_AEs[~np.isnan(binned_AEs)]
        
        bin_maes.update({labels[i]: [x, binned_AEs]})
        
        ax.plot(x, binned_AEs, c = colours[i], marker = markers[i], markerfacecolor = "#f0f0f0",
                label = labels[i])

    bin_width = (bin_edges[-1] - bin_edges[0])/nbins
    axtwin.hist(ys, color = '#2166ac', bins = nbins, width=0.85 * bin_width,
                edgecolor="#878787", linewidth= .5)
    ax.set_xlabel(r"Experimental $\Delta$H (kJ/mol H$_2$)")
    axtwin.set_ylabel(r"Testset counts")
    axtwin.tick_params(axis='y', colors='#2166ac')
    axtwin.yaxis.label.set_color('#2166ac')
    
    ax.set_zorder(axtwin.get_zorder()+1)
    ax.patch.set_visible(False)
    
    ax.annotate(
    '',
    xy=(52, 25), xycoords='data',
    xytext=(61, 25),
    arrowprops=dict(arrowstyle="->"))
    
    ax.annotate(
    '',
    xy=(45, 25), xycoords='data',
    xytext=(36, 25),
    arrowprops=dict(arrowstyle="->"))
    
    if split == 'test':
        ax.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Test}}$ within bin")
    elif split == "train":
        ax.set_ylabel(r"$\langle$MAE$\rangle_{\mathrm{Train}}$ within bin")
    ax.tick_params(axis='y')
    # ax.yaxis.label.set_color('#b2182b')

    ax.legend(loc = 'upper left', framealpha = 0.2, title_fontproperties = {'weight': 'semibold'})
    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()
    return bin_maes