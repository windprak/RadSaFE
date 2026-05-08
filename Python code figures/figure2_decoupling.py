"""
figure2_decoupling.py
Figure 2: Safety and accuracy decouple across deployment conditions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


class Figure2:
    MAIN_CSV = './Main_results.csv'
    OUT_PNG = './figs/figure2_decoupling.png'

    CONDITIONS = ['Closed-book', 'Clean evidence', 'Conflict evidence',
                  'Standard RAG', 'RaR', 'Max context']

    COND_DISPLAY = {
        'Closed-book': 'Closed-book',
        'Clean evidence': 'Clean evidence',
        'Conflict evidence': 'Conflict evidence',
        'Standard RAG': 'Standard RAG',
        'RaR': 'Agentic RAG',
        'Max context': 'Max context',
    }

    COND_COLOR = {
        'Closed-book':       '#525252',
        'Clean evidence':    '#2166AC',
        'Conflict evidence': '#67A9CF',
        'Standard RAG':      '#F4A582',
        'RaR':               '#D6604D',
        'Max context':       '#B2182B',
    }

    def __init__(self):
        self._set_rcparams()
        self.df = self._load()
        self.df_avg = self.df[self.df['Model'] == 'Average of all models'].copy()
        self.df_ind = self.df[self.df['Model'] != 'Average of all models'].copy()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 14,
            'axes.labelsize': 15.5,
            'axes.titlesize': 15.5,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.1,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'axes.grid': False,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

    def _load(self):
        df = pd.read_csv(self.MAIN_CSV)
        df['acc'] = df['Accuracy (mean ± std [95% CI])'].apply(
            lambda s: float(str(s).split('±')[0].strip()))
        df = df.rename(columns={
            'High risk error (rate)': 'high_risk',
            'Unsafe answer (rate)': 'unsafe',
            'Contradiction (rate)': 'contradiction',
            'Dangerous overconfidence (rate)': 'dang_oc',
            'Mean confidence': 'mean_conf',
            'Mean latency (seconds)': 'latency',
        })
        return df

    @staticmethod
    def add_panel_label(ax, letter, title='', x=-0.18, y=1.04):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='bottom', ha='left')
        if title:
            ax.text(x + 0.08, y, title, transform=ax.transAxes,
                    fontsize=14, fontweight='normal', va='bottom', ha='left')

    def build(self):
        self.fig = plt.figure(figsize=(17, 18))

        # Legend strip at top
        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.998, bottom=0.965)
        # Three rows: 2 wide / 3 medium / 2 wide
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.945, bottom=0.685, wspace=0.28)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.07, right=0.98,
                                      top=0.625, bottom=0.365, wspace=0.36)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.305, bottom=0.045,
                                      wspace=0.28, width_ratios=[1.05, 1])

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 2]))
        self._panel_f(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_g(self.fig.add_subplot(gs_r3[0, 1]))
        return self

    def _draw_legend(self, ax):
        ax.axis('off')
        items = [
            Line2D([0], [0], marker='o', color=self.COND_COLOR[c],
                   mfc=self.COND_COLOR[c], mec='white', ms=11, lw=0,
                   label=self.COND_DISPLAY[c])
            for c in self.CONDITIONS
        ]
        ax.legend(handles=items, loc='center', ncol=6, frameon=False,
                  fontsize=14, handletextpad=0.4, columnspacing=2.0)

    # ----- Panel a: accuracy vs high-risk error scatter -----
    def _panel_a(self, ax):
        # shaded "safe deployment" region
        ax.add_patch(Rectangle((90, 0), 12, 5,
                               facecolor='#e8f0e0', edgecolor='#86a96e',
                               alpha=0.5, zorder=0, lw=1.0, ls='--'))

        for cond in self.CONDITIONS:
            sub = self.df_ind[self.df_ind['Condition'] == cond]
            ax.scatter(sub['acc'], sub['high_risk'],
                       s=44, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.6,
                       alpha=0.80, zorder=3)
            # condition centroid
            avg = self.df_avg[self.df_avg['Condition'] == cond].iloc[0]
            ax.scatter(avg['acc'], avg['high_risk'], s=340, marker='*',
                       c=self.COND_COLOR[cond], edgecolors='black',
                       linewidths=1.4, zorder=5)

        ax.text(96, 2.5, 'Safe deployment\nregion',
                fontsize=12, color='#3c6e2c', ha='center', va='center',
                style='italic', zorder=6,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5))

        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('High-risk error rate (%)')
        ax.set_xlim(15, 102)
        ax.set_ylim(-0.8, 30)
        ax.invert_yaxis()
        self.add_panel_label(ax, 'a',
                             'Curated evidence collapses high-risk errors')

    # ----- Panel b: accuracy vs dangerous overconfidence -----
    def _panel_b(self, ax):
        ax.add_patch(Rectangle((90, 0), 12, 5,
                               facecolor='#e8f0e0', edgecolor='#86a96e',
                               alpha=0.5, zorder=0, lw=1.0, ls='--'))

        for cond in self.CONDITIONS:
            sub = self.df_ind[self.df_ind['Condition'] == cond]
            ax.scatter(sub['acc'], sub['dang_oc'],
                       s=44, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.6,
                       alpha=0.80, zorder=3)
            avg = self.df_avg[self.df_avg['Condition'] == cond].iloc[0]
            ax.scatter(avg['acc'], avg['dang_oc'], s=340, marker='*',
                       c=self.COND_COLOR[cond], edgecolors='black',
                       linewidths=1.4, zorder=5)

        ax.text(96, 2.5, 'Safe deployment\nregion',
                fontsize=12, color='#3c6e2c', ha='center', va='center',
                style='italic', zorder=6,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5))

        # Annotate the agentic RAG vs standard RAG centroid relationship
        rag_avg = self.df_avg[self.df_avg['Condition'] == 'Standard RAG'].iloc[0]
        rar_avg = self.df_avg[self.df_avg['Condition'] == 'RaR'].iloc[0]
        ax.annotate('', xy=(rar_avg['acc'], rar_avg['dang_oc']),
                    xytext=(rag_avg['acc'], rag_avg['dang_oc']),
                    arrowprops=dict(arrowstyle='->', color='black',
                                    lw=2.0, shrinkA=10, shrinkB=10),
                    zorder=4)
        ax.text(40, 30,
                'Agentic RAG: ↑ accuracy,\nbut ↑ dangerous overconf.',
                fontsize=11.5, ha='left', va='center',
                color='#222', style='italic',
                bbox=dict(facecolor='white', edgecolor='#888',
                          alpha=0.92, pad=3, lw=0.6))
        ax.annotate('', xy=((rag_avg['acc'] + rar_avg['acc']) / 2,
                            (rag_avg['dang_oc'] + rar_avg['dang_oc']) / 2 + 1.5),
                    xytext=(54, 30),
                    arrowprops=dict(arrowstyle='-', color='#888',
                                    lw=0.8, ls='--'),
                    zorder=3)

        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Dangerous overconfidence rate (%)')
        ax.set_xlim(15, 102)
        ax.set_ylim(-1.5, 50)
        ax.invert_yaxis()
        self.add_panel_label(ax, 'b',
                             'Confidence-laden errors are decoupled from accuracy')

    # ----- helper to draw paired connectors -----
    def _draw_pairs(self, ax, cond_from, cond_to, x_metric, y_metric,
                    color_from, color_to):
        df_from = self.df_ind[self.df_ind['Condition'] == cond_from
                              ].set_index('Model')
        df_to = self.df_ind[self.df_ind['Condition'] == cond_to
                            ].set_index('Model')
        common = df_from.index.intersection(df_to.index)
        for m in common:
            x0, y0 = df_from.loc[m, x_metric], df_from.loc[m, y_metric]
            x1, y1 = df_to.loc[m, x_metric], df_to.loc[m, y_metric]
            ax.plot([x0, x1], [y0, y1], color='#999', lw=0.7,
                    alpha=0.5, zorder=2)
            arr = FancyArrowPatch((x0, y0), (x1, y1),
                                  arrowstyle='->,head_length=4,head_width=3',
                                  color='#666', lw=0.0, alpha=0.0,
                                  zorder=2)
            ax.add_patch(arr)
        # endpoint dots
        ax.scatter(df_from.loc[common, x_metric],
                   df_from.loc[common, y_metric],
                   s=36, c=color_from, edgecolors='white',
                   linewidths=0.6, alpha=0.85, zorder=3)
        ax.scatter(df_to.loc[common, x_metric],
                   df_to.loc[common, y_metric],
                   s=36, c=color_to, edgecolors='white',
                   linewidths=0.6, alpha=0.85, zorder=3)

        # centroid arrow
        x0c = df_from.loc[common, x_metric].mean()
        y0c = df_from.loc[common, y_metric].mean()
        x1c = df_to.loc[common, x_metric].mean()
        y1c = df_to.loc[common, y_metric].mean()
        ax.annotate('', xy=(x1c, y1c), xytext=(x0c, y0c),
                    arrowprops=dict(arrowstyle='-|>', color='black',
                                    lw=2.4, mutation_scale=22),
                    zorder=6)
        ax.scatter([x0c], [y0c], s=240, marker='*',
                   c=color_from, edgecolors='black',
                   linewidths=1.4, zorder=7)
        ax.scatter([x1c], [y1c], s=240, marker='*',
                   c=color_to, edgecolors='black',
                   linewidths=1.4, zorder=7)

    def _panel_c(self, ax):
        self._draw_pairs(ax, 'Closed-book', 'Clean evidence',
                         'acc', 'high_risk',
                         self.COND_COLOR['Closed-book'],
                         self.COND_COLOR['Clean evidence'])
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('High-risk error rate (%)')
        ax.set_xlim(15, 100)
        ax.set_ylim(-0.5, 28)
        ax.invert_yaxis()
        ax.text(0.04, 0.06,
                r'closed-book $\rightarrow$ clean evidence',
                transform=ax.transAxes, fontsize=12, color='#222')
        self.add_panel_label(ax, 'c',
                             'Universal shift toward safe corner')

    def _panel_d(self, ax):
        self._draw_pairs(ax, 'Closed-book', 'Standard RAG',
                         'acc', 'high_risk',
                         self.COND_COLOR['Closed-book'],
                         self.COND_COLOR['Standard RAG'])
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('High-risk error rate (%)')
        ax.set_xlim(15, 100)
        ax.set_ylim(-0.5, 28)
        ax.invert_yaxis()
        ax.text(0.04, 0.06,
                r'closed-book $\rightarrow$ standard RAG',
                transform=ax.transAxes, fontsize=12, color='#222')
        self.add_panel_label(ax, 'd',
                             'Standard RAG: accuracy gain without safety gain')

    def _panel_e(self, ax):
        self._draw_pairs(ax, 'Standard RAG', 'RaR',
                         'acc', 'dang_oc',
                         self.COND_COLOR['Standard RAG'],
                         self.COND_COLOR['RaR'])
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Dangerous overconfidence rate (%)')
        ax.set_xlim(35, 100)
        ax.set_ylim(-1.5, 50)
        ax.invert_yaxis()
        ax.text(0.04, 0.06,
                r'standard RAG $\rightarrow$ agentic RAG',
                transform=ax.transAxes, fontsize=12, color='#222')
        self.add_panel_label(ax, 'e',
                             'Agentic RAG often increases overconfidence')

    # ----- Panel f: parallel coordinates of safety profile -----
    def _panel_f(self, ax):
        axes_labels = ['Accuracy', '1 - High-risk', '1 - Unsafe',
                       '1 - Contradiction', '1 - Dangerous\noverconf.']
        n_axes = len(axes_labels)
        x_pos = np.arange(n_axes)

        for cond in self.CONDITIONS:
            row = self.df_avg[self.df_avg['Condition'] == cond].iloc[0]
            vals = [
                row['acc'],
                100 - row['high_risk'],
                100 - row['unsafe'],
                100 - row['contradiction'],
                100 - row['dang_oc'],
            ]
            lw = 3.4 if cond in ('Clean evidence', 'Conflict evidence') else 2.0
            alpha = 1.0 if cond in ('Clean evidence', 'Conflict evidence') else 0.85
            ax.plot(x_pos, vals, '-o', color=self.COND_COLOR[cond],
                    lw=lw, ms=10, mec='white', mew=1.0, alpha=alpha,
                    zorder=4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(axes_labels, fontsize=13, ha='center')
        ax.set_ylabel('Score (%)')
        ax.set_ylim(70, 102)
        ax.set_yticks([70, 80, 90, 100])
        for x in x_pos:
            ax.axvline(x, color='#dddddd', lw=0.8, zorder=1)
        ax.set_xlim(-0.3, n_axes - 0.7)

        # No legend - using top legend instead
        self.add_panel_label(ax, 'f',
                             'Curated evidence dominates the full safety profile')

    # ----- Panel g: latency vs accuracy bubble, size = 1/dang_oc -----
    def _panel_g(self, ax):
        # per (model, condition) point: x=latency, y=acc, size~1/(dang_oc+1)
        for cond in self.CONDITIONS:
            sub = self.df_ind[self.df_ind['Condition'] == cond].copy()
            sizes = 1500.0 / (sub['dang_oc'] + 3.0)  # bigger = safer
            ax.scatter(sub['latency'], sub['acc'],
                       s=sizes, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.7,
                       alpha=0.50, zorder=3)
            avg = self.df_avg[self.df_avg['Condition'] == cond].iloc[0]
            ax.scatter(avg['latency'], avg['acc'], s=420, marker='*',
                       c=self.COND_COLOR[cond], edgecolors='black',
                       linewidths=1.5, zorder=6)

        # Custom label placement to avoid overlap (offset in points)
        label_pos = {
            'Closed-book':       (18.0, 73.5,  -85, 22,  'right'),
            'Clean evidence':    (17.3, 94.1,  -25, 18,  'right'),
            'Conflict evidence': (17.5, 92.5,  60,  -10, 'left'),
            'Standard RAG':      (18.4, 76.0,  85,  18,  'left'),
            'RaR':               (19.9, 78.1,  95,  -2,  'left'),
            'Max context':       (27.0, 74.0,  70,  -22, 'left'),
        }
        for cond in self.CONDITIONS:
            x, y, dx, dy, ha = label_pos[cond]
            ax.annotate(self.COND_DISPLAY[cond],
                        xy=(x, y),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        color=self.COND_COLOR[cond], ha=ha, va='center',
                        arrowprops=dict(arrowstyle='-', color=self.COND_COLOR[cond],
                                        lw=0.8, alpha=0.6))

        ax.set_xscale('log')
        ax.set_xlabel('Mean latency (s, log scale)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlim(0.2, 280)
        ax.set_ylim(15, 102)

        # bubble-size legend in upper-left where space exists
        for s_val, ref_val in [(1500.0 / 23, 20),
                               (1500.0 / 13, 10),
                               (1500.0 / 6,  3)]:
            ax.scatter([], [], s=s_val, c='#bbbbbb', edgecolors='white',
                       alpha=0.7, label=f'≈ {ref_val}%')
        ax.legend(loc='lower right', frameon=False, fontsize=11,
                  scatterpoints=1, labelspacing=1.4, handletextpad=1.2,
                  borderpad=0.6,
                  title='Dangerous overconf.\n(marker size)',
                  title_fontsize=11)
        self.add_panel_label(ax, 'g',
                             'Latency cost without proportional accuracy or safety gain')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')
        print(f'Saved: {self.OUT_PNG}')


if __name__ == '__main__':
    Figure2().build().save()
