import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    for patch in bp['boxes']:
        patch.set(facecolor = color)
    plt.setp(bp['whiskers'], color='cornflowerblue')
    plt.setp(bp['caps'], color='steelblue')
    plt.setp(bp['medians'], color='dodgerblue')

model_names = ['Initial', 'SyN', 'SYMNet', 'VoxelMorph', 'TransMorph', 'TransMorph_Uncertain']
results_dir = ['./Quantitative_Results/initial_ncc_1_diffusion_1_dsc_1.csv',
               './Quantitative_Results/ants_ACDC.csv',
               './SYMNet/Code/Quantitative_Results/SYMNet.csv',
               './VoxelMorph/Quantitative_Results/VoxelMorph2_ncc_1_diffusion_1_dsc_1.csv',
               './Quantitative_Results/TransMorphTVF_ncc_1_diffusion_1_dsc_1.csv',
               './Quantitative_Results/TransMorphTVFUncertBayesEntropy_ncc_1_diffusion_1_nll_1_dsc_1.csv',]
# clus_1:RV; clus_2:MYO;clus_3:LV
for idx, model in enumerate(model_names):
    dataf = pd.read_csv(results_dir[idx])
    RV  = dataf['clus_1'].to_numpy()
    MYO = dataf['clus_2'].to_numpy()
    LV  = dataf['clus_3'].to_numpy()
    mdsc = (RV+MYO+LV)/3.
    NDV  = dataf['NDV'].to_numpy()
    NPDJ = dataf['NPDJ'].to_numpy()

    print('{} - \n'
          'RV: {:.3f}$\pm${:.3f}, MYO: {:.3f}$\pm${:.3f}, LV: {:.3f}$\pm${:.3f}, avg.dsc: {:.3f}$\pm${:.3f}\n'
          'NDV: {:.3f}$\pm${:.3f}, NPDJ: {:.3f}$\pm${:.3f}'.format(model, RV.mean(), RV.std(), MYO.mean(), MYO.std(), LV.mean(), LV.std(), mdsc.mean(),
                                                               mdsc.std(), NDV.mean(), NDV.std(), NPDJ.mean(), NPDJ.std()))

    if "UncertBayes" in results_dir[idx]:
        outstruct = ['LV', 'RV', 'MYO']
        cc_epi_RV  = dataf['cc_epi_1'].to_numpy()
        cc_epi_MYO = dataf['cc_epi_2'].to_numpy()
        cc_epi_LV  = dataf['cc_epi_3'].to_numpy()
        cc_epi_avg = (cc_epi_RV + cc_epi_MYO + cc_epi_LV) / 3.
        cc_ale_RV  = dataf['cc_ale_1'].to_numpy()
        cc_ale_MYO = dataf['cc_ale_2'].to_numpy()
        cc_ale_LV  = dataf['cc_ale_3'].to_numpy()
        cc_ale_avg = (cc_ale_RV + cc_ale_MYO + cc_ale_LV) / 3.
        cc_two_RV  = dataf['cc_two_1'].to_numpy()
        cc_two_MYO = dataf['cc_two_2'].to_numpy()
        cc_two_LV  = dataf['cc_two_3'].to_numpy()
        cc_two_avg = (cc_two_RV+cc_two_MYO+cc_two_LV)/3.
        cc_tra_RV = dataf['cc_tra_1'].to_numpy()
        cc_tra_MYO = dataf['cc_tra_2'].to_numpy()
        cc_tra_LV = dataf['cc_tra_3'].to_numpy()
        cc_tra_avg = (cc_tra_RV + cc_tra_MYO + cc_tra_LV) / 3.
        cc_app_RV = dataf['cc_app_1'].to_numpy()
        cc_app_MYO = dataf['cc_app_2'].to_numpy()
        cc_app_LV = dataf['cc_app_3'].to_numpy()
        cc_app_avg = (cc_app_RV + cc_app_MYO + cc_app_LV) / 3.
        cc_epi = np.stack([cc_epi_LV, cc_epi_RV, cc_epi_MYO], axis=1)
        cc_ale = np.stack([cc_ale_LV, cc_ale_RV, cc_ale_MYO], axis=1)
        cc_two = np.stack([cc_two_LV, cc_two_RV, cc_two_MYO], axis=1)

        print('{:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}\n'
              '{:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}\n'
              '{:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}\n'
              '{:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}\n'
              '{:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}\n'.format(cc_epi_LV.mean(), cc_epi_LV.std(),
                                                                                                       cc_epi_RV.mean(), cc_epi_RV.std(),
                                                                                                       cc_epi_MYO.mean(), cc_epi_MYO.std(),
                                                                                                       cc_epi_avg.mean(),cc_epi_avg.std(),
                                                                                                       cc_ale_LV.mean(), cc_ale_LV.std(),
                                                                                                       cc_ale_RV.mean(), cc_ale_RV.std(),
                                                                                                       cc_ale_MYO.mean(), cc_ale_MYO.std(),
                                                                                                       cc_ale_avg.mean(), cc_ale_avg.std(),
                                                                                                       cc_two_LV.mean(), cc_two_LV.std(),
                                                                                                       cc_two_RV.mean(), cc_two_RV.std(),
                                                                                                       cc_two_MYO.mean(),cc_two_MYO.std(),
                                                                                                       cc_two_avg.mean(),cc_two_avg.std(),
                                                                                                       cc_tra_LV.mean(),cc_tra_LV.std(),
                                                                                                       cc_tra_RV.mean(),cc_tra_RV.std(),
                                                                                                       cc_tra_MYO.mean(),cc_tra_MYO.std(),
                                                                                                       cc_tra_avg.mean(),cc_tra_avg.std(),
                                                                                                       cc_app_LV.mean(),cc_app_LV.std(),
                                                                                                       cc_app_RV.mean(),cc_app_RV.std(),
                                                                                                       cc_app_MYO.mean(),cc_app_MYO.std(),
                                                                                                       cc_app_avg.mean(),cc_app_avg.std(),
                                                                                                       ))

        flierprops = dict(marker='o', markerfacecolor='cornflowerblue', markersize=2, linestyle='none',
                          markeredgecolor='grey')
        meanprops = {"markerfacecolor": "sandybrown", "markeredgecolor": "chocolate"}
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        spacing_factor = 4
        showmeans = False
        sep = 1
        affine = plt.boxplot(cc_epi, labels=outstruct,
                             positions=np.array(range(len(outstruct))) * spacing_factor - sep, widths=0.6,
                             showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
        nifty = plt.boxplot(cc_ale, labels=outstruct,
                            positions=np.array(range(len(outstruct))) * spacing_factor - sep * 0, widths=0.6,
                            showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
        syn = plt.boxplot(cc_two, labels=outstruct,
                          positions=np.array(range(len(outstruct))) * spacing_factor + sep, widths=0.6,
                          showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
        set_box_color(affine, 'plum')  # colors are from http://colorbrewer2.org/
        set_box_color(nifty, 'slateblue')
        set_box_color(syn, 'tan')
        plt.grid(linestyle='--', linewidth=1)
        plt.plot([], c='plum', label='Epistemic Uncert.')
        plt.plot([], c='slateblue', label='Aleatoric Uncert.')
        plt.plot([], c='tan', label='Combined Uncert.')
        font = font_manager.FontProperties(family='Cambria',
                                           style='normal', size=10)
        leg = ax.legend(prop=font)
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        minor_ticks = np.arange(-10.8, len(outstruct) * spacing_factor, 0.8)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(np.arange(0, 1.05, 0.2))
        ax.set_yticks(np.arange(-0.05, 1.05, 0.05), minor=True)
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        plt.xticks(range(0, len(outstruct) * spacing_factor, spacing_factor), outstruct, fontsize=14, )
        plt.yticks(fontsize=20)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Cambria")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Cambria")
        plt.xlim(-2, len(outstruct) * spacing_factor - 2)
        plt.ylim(-0.05, 0.85)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.4)
        plt.close()
