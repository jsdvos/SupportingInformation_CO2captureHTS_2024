import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import colors

labels_latex = {
        'rho': r'$\rho$',
        'di': r'$D_i$',
        'dif': r'$D_{if}$',
        'df': r'$D_f$',
        'av_v': r'$VPOV$',
        'av_m': r'$GPOV$',
        'asa_v': r'$VSA$',
        'asa_m': r'$GSA$',
        'nav_v': r'$nVPOV$',
        'nav_m': r'$nGPOV$',
        'nasa_v': r'$nVSA$',
        'nasa_m': r'$nGSA$',
        'lig_I': r'$I$',
        'lig_T': r'$T$',
        'lig_X': r'$\chi$',
        'lig_S': r'$S$',
        'lig_Z': r'$Z$',
        'lig_a': r'$\alpha$',
        'linkresc_I': r'$I$',
        'linkresc_T': r'$T$',
        'linkresc_X': r'$\chi$',
        'linkresc_S': r'$S$',
        'linkresc_Z': r'$Z$',
        'linkresc_a': r'$\alpha$',
        'func_I': r'$I$',
        'func_T': r'$T$',
        'func_X': r'$\chi$',
        'func_S': r'$S$',
        'func_Z': r'$Z$',
        'func_a': r'$\alpha$',
        }
for p in ['I', 'T', 'X', 'S', 'Z', 'a']:
    for domain in ['lig', 'linkresc', 'func']:
        for depth in ['0', '1', '2', '3']:
            for method in ['prod', 'diff']:
                domain_ = {'lig': 'linkage', 'linkresc': 'linker', 'func': 'func'}[domain]
                p_ = {'I': 'I', 'T': 'T', 'X': r'\chi', 'S': 'S', 'Z': 'Z', 'a': r'\alpha'}[p]
                labels_latex['_'.join([domain, method, p, depth])] = r'$^\mathregular{{{}}}{{{}}}_{}^\mathregular{{{}}}$'.format(domain_, p_, depth, method)

def combine_importance_values(importance_values):
    keys = importance_values.keys()
    for env in ['linkresc', 'lig', 'func']:
        for prop in ['I', 'T', 'X', 'S', 'Z', 'a']:
            result = 0.0
            for method in ['diff', 'prod']:
                for depth in [0, 1, 2, 3]:
                    key = '{}_{}_{}_{}'.format(env, method, p, depth)
                    if not key in keys: continue
                    result += importance_values.pop(key)
            importance_values['{}_{}'.format(env, prop)] = result

    result = {'geo': {}, 'linkresc': {}, 'lig': {}, 'func': {}}
    for key, value in importance_values.items():
        if key.startswith('linkresc'):
            result['linkresc'][key] = value
        elif key.startswith('lig'):
            result['lig'][key] = value
        elif key.startswith('func'):
            result['func'][key] = value
        else:
            result['geo'][key] = value
    return result

def plot_pie(importance_values, fig_name):
    def get_cmap(color):
        vals = np.ones((256, 3))
        r, g, b = colors.to_rgb(color)
        vals[:, 0] = np.linspace(1, r, 256)
        vals[:, 1] = np.linspace(1, g, 256)
        vals[:, 2] = np.linspace(1, b, 256)
        return colors.ListedColormap(vals)

    color_index = 0
    sizes = []
    sizes_domain = []
    cs = []
    labels = []
    tot = sum(sum(importance_values[x].values()) for x in ['geo', 'lig', 'linkresc', 'func'])
    for key in ['geo', 'lig', 'linkresc', 'func']:
        color = 'C{}'.format(color_index)
        color_index += 1
        cmap = get_cmap(color)
        sizes_domain.append(0.0)
        for i, prop in enumerate(sorted(importance_values[key].keys(), reverse = True, key = lambda e: importance_values[key][e])):
            value = importance_values[key][prop]
            sizes.append(value/tot)
            sizes_domain[-1] += value/tot
            if value/tot > 0.03:
                labels.append(labels_latex[prop])
            else:
                labels.append('')
            cs.append(cmap((float(len(importance_values[key])-i)/len(importance_values[key]))))

    fig, ax = plt.subplots()
    ax.pie(sizes, colors = cs, labels = labels, wedgeprops = {'linewidth': 0.1, 'edgecolor': 'k'})
    ax.pie(sizes_domain, colors = [(.0,.0,.0,.0)]*4, wedgeprops = {'linewidth': 1, 'edgecolor': 'k'})
    plt.savefig(fig_name, bbox_inches = 'tight')
    plt.clf()

if __name__ == '__main__':
    # Define data features
    features = pd.read_csv('../data/input/features.csv', sep = ';')
    features['mass'] = features['rho']*features['vol']
    for feature in ['asa', 'av', 'nasa', 'nav']:
        features[feature + '_m'] = features[feature]/features['mass']
        features[feature + '_v'] = features[feature]/features['vol']
    rac_features = ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3']
    for feature in rac_features:
        features['linkresc_' + feature] = np.where(features['link_N_start'] != 0, features['link_' + feature]*features['link_N_start']/features['link_N_part'], 0.0)

    # Iterate over models
    for algorithm in ['KRR', 'SVR', 'MLP', 'GB', 'RF']:
        for run in [0, 1, 2, 3, 4]:
            # Initiate Explainer
            model = '{}{}'.format(algorithm, run)
            df = pd.read_csv('../data/shap/shap_{}.csv'.format(model))
            struct = df['struct'].values
            feats = features[[x in struct for x in features['struct'].values]]
            df.drop(['Unnamed: 0', 'struct'], axis = 1, inplace = True)
            explainer = shap.Explanation(
                    values = df.values,
                    feature_names = [labels_latex[key] for key in df.columns],
                    data = feats[df.columns].values
                    )
            #### Create SHAP plots
            # Plot 1: pie chart
            importance_values = {}
            for column in df.columns:
                importance_values[column] = np.mean(np.abs(df[column].values))
            importance_values = combine_importance_values(importance_values)
            plot_pie(importance_values, '../figs/shap/pie/{}.pdf'.format(model))
            # Plot 2: scatter plot with matplotlib
            x = feats['rho'].values/0.16267262832702767 # conversion to kg/m3
            y = df['rho'].values
            fig, ax = plt.subplots()
            cbar = ax.hexbin(x, y, mincnt = 1, norm = colors.LogNorm(vmin = 1.0), cmap = shap.plots.colors.red_blue)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.colorbar(cbar)
            plt.xlabel(r'$\rho$ [$\mathregular{kg/m^3}$]')
            plt.ylabel(r'SHAP value for $\rho$ [-]')
            plt.savefig('../figs/shap/scatter/{}.pdf'.format(model), bbox_inches = 'tight')
            plt.clf()
            # Plot 3: beeswarm plot
            shap.plots.beeswarm(explainer, show = False)
            plt.savefig('../figs/shap/beeswarm/{}.pdf'.format(model), bbox_inches = 'tight')
            plt.clf()






