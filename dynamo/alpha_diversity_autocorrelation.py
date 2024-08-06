import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class AlphaDiversityAutocorrelation:

    @staticmethod
    def calculate_and_plot_acf(ts, subject):
        # Calculate acf and pacf functions
        acf_vals, acf_ci, acf_qstat, acf_pvalues = sm.tsa.stattools.acf(ts, nlags=70, fft=False, alpha=0.05, qstat=True)
        pacf_vals, pacf_ci = sm.tsa.stattools.pacf(ts, nlags=70, alpha=0.05)

        # Center the confidence intervals around zero
        centered_acf_ci = acf_ci - np.stack([acf_vals, acf_vals], axis=1)
        centered_pacf_ci = pacf_ci - np.stack([pacf_vals, pacf_vals], axis=1)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        markerline1, stemlines1, baseline1 = axes[0].stem(acf_vals, linefmt='black', markerfmt='o')
        markerline1.set_markerfacecolor('black')
        markerline1.set_markersize(5)
        markerline1.set_markeredgewidth(0)
        stemlines1.set_linewidth(.7)
        baseline1.set_linewidth(3)
        axes[0].fill_between(range(0, 71), centered_acf_ci[:, 0], centered_acf_ci[:, 1], alpha=.2)
        axes[0].set_xlabel('lag [day]', size=14)
        axes[0].set_ylabel('ACF', size=14)
        axes[0].tick_params(axis='both', which='major', labelsize=10)

        markerline2, stemlines2, baseline2 = axes[1].stem(pacf_vals, linefmt='black', markerfmt='o')
        markerline2.set_markerfacecolor('black')
        markerline2.set_markersize(5)
        markerline2.set_markeredgewidth(0)
        stemlines2.set_linewidth(.7)
        baseline2.set_linewidth(3)
        axes[1].fill_between(range(0, 71), centered_pacf_ci[:, 0], centered_pacf_ci[:, 1], alpha=.2)
        axes[1].set_xlabel('lag [day]', size=14)
        axes[1].set_ylabel('PACF', size=14)
        axes[1].tick_params(axis='both', which='major', labelsize=10)

        fig.suptitle(f'{subject}', size=12)
        plt.tight_layout()
        # plt.savefig(f'/storage/zkarwowska/microbiome-dynamics-preprint/plots/whole_community/autocorrelation/{subject}_acf_shannon.png', dpi=300)
        plt.show()