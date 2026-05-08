"""
figure6_ensembles.py
Figure 6: Ensembles do not produce safety; synchronized failure is a new failure mode.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Figure6:
    ENS_CSV = './ensemble_results.csv'
    MAIN_CSV = './Main_results.csv'
    OUT_PNG = './figs/figure6_ensembles.png'

    CONDITIONS = ['Closed-book', 'Conflict evidence', 'Standard RAG']

    COND_DISPLAY = {
        'Closed-book': 'Closed-book',
        'Conflict evidence': 'Conflict evidence',
        'Standard RAG': 'Standard RAG',
    }

    COND_COLOR = {
        'Closed-book':       '#525252',
        'Conflict evidence': '#67A9CF',
        'Standard RAG':      '#F4A582',
    }

    ENSEMBLE_ORDER = ['Dense Mid', 'Frontier', 'Qwen scale', 'Cross scale']

    ENS_COLOR = {
        'Dense Mid':   '#1B7A78',
        'Frontier':    '#1F4E89',
        'Qwen scale':  '#7E459C',
        'Cross scale': '#A65628',
    }

    def __init__(self):
        self._set_rcparams()
        self.df_ens = pd.read_csv(self.ENS_CSV)
        self.df_main = self._load_main()
        self._parse_ensemble_columns()
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

    def _load_main(self):
        df = pd.read_csv(self.MAIN_CSV)
        df = df[df['Model'] != 'Average of all models'].copy()
        df['acc'] = df['Accuracy (mean ± std [95% CI])'].apply(
            lambda s: float(str(s).split('±')[0].strip()))
        df = df.rename(columns={
            'High risk error (rate)': 'high_risk',
            'Unsafe answer (rate)': 'unsafe',
            'Contradiction (rate)': 'contradiction',
            'Dangerous overconfidence (rate)': 'dang_oc',
        })
        return df

    def _parse_ensemble_columns(self):
        df = self.df_ens
        df['acc'] = df['Accuracy (mean ± std [95% CI])'].apply(
            lambda s: float(str(s).split('±')[0].strip()))
        df.rename(columns={
            'High risk error (rate)': 'high_risk',
            'Unsafe answer (rate)': 'unsafe',
            'Contradiction (rate)': 'contradiction',
            'Dangerous overconfidence (rate)': 'dang_oc',
            'Synchronized failure (rate)': 'sync_fail',
            'Ensemble name': 'ensemble',
        }, inplace=True)

    def _member_metric(self, ensemble, condition, metric):
        """Return the per-member values of metric for the 3 members of
        ensemble under the given condition (from Main_results)."""
        row = self.df_ens[(self.df_ens['ensemble'] == ensemble) &
                          (self.df_ens['Condition'] == condition)].iloc[0]
        members = [row['Member model 1'], row['Member model 2'],
                   row['Member model 3']]
        vals = []
        for m in members:
            sub = self.df_main[(self.df_main['Model'] == m) &
                               (self.df_main['Condition'] == condition)]
            if len(sub):
                vals.append(sub[metric].values[0])
        return vals

    @staticmethod
    def add_panel_label(ax, letter, title='', x=-0.18, y=1.04):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='bottom', ha='left')
        if title:
            ax.text(x + 0.08, y, title, transform=ax.transAxes,
                    fontsize=14, fontweight='normal', va='bottom', ha='left')

    def build(self):
        self.fig = plt.figure(figsize=(17, 21))

        # Top legend strip - more breathing room at top
        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.985, bottom=0.948)
        # Row 1: panels a, b, c — three ensemble-vs-member panels
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.07, right=0.98,
                                      top=0.918, bottom=0.685, wspace=0.34)
        # Row 2: panels d, e
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.640, bottom=0.378,
                                      wspace=0.32, width_ratios=[1.2, 1])
        # Row 3: panels f, g
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.310, bottom=0.045,
                                      wspace=0.32, width_ratios=[1, 1.25])

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r1[0, 2]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_f(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_g(self.fig.add_subplot(gs_r3[0, 1]))
        return self

    def _draw_legend(self, ax):
        ax.axis('off')
        cond_items = [
            Line2D([0], [0], marker='*', color=self.COND_COLOR[c],
                   mfc=self.COND_COLOR[c], mec='black', ms=16, mew=1.0,
                   lw=0, label=self.COND_DISPLAY[c])
            for c in self.CONDITIONS
        ]
        marker_items = [
            Line2D([0], [0], marker='*', mfc='white', mec='black', ms=16,
                   mew=1.2, lw=0, label='Ensemble'),
            Line2D([0], [0], marker='o', mfc='#aaaaaa', mec='white', ms=10,
                   lw=0, label='Individual member'),
        ]
        leg1 = ax.legend(handles=cond_items, loc='center left',
                         bbox_to_anchor=(0.0, 0.5),
                         ncol=3, frameon=False, fontsize=13.5,
                         handletextpad=0.5, columnspacing=2.0,
                         title='Condition', title_fontsize=13.5)
        ax.add_artist(leg1)
        ax.legend(handles=marker_items, loc='center right',
                  bbox_to_anchor=(1.0, 0.5),
                  ncol=2, frameon=False, fontsize=13.5,
                  handletextpad=0.5, columnspacing=1.5,
                  title='Marker', title_fontsize=13.5)

    # ----- Helper: ensemble vs member dotplot -----
    def _draw_ens_vs_member(self, ax, metric, ylabel, ylim, lower_better):
        """For each (ensemble, condition), plot 3 member dots and 1 ensemble star.
        Group: x-axis grouped by ensemble, with 3 condition slots per ensemble."""
        n_ens = len(self.ENSEMBLE_ORDER)
        n_cond = len(self.CONDITIONS)
        rng = np.random.default_rng(7)

        for i_e, ens in enumerate(self.ENSEMBLE_ORDER):
            for i_c, cond in enumerate(self.CONDITIONS):
                # x-position: group by ensemble, condition slots within
                x = i_e + (i_c - (n_cond - 1) / 2) * 0.22
                # Ensemble value
                row = self.df_ens[(self.df_ens['ensemble'] == ens) &
                                  (self.df_ens['Condition'] == cond)].iloc[0]
                v_ens = row[metric]
                # Member values
                m_vals = self._member_metric(ens, cond, metric)

                # Vertical "tick" connecting members
                if m_vals:
                    ax.plot([x, x], [min(m_vals), max(m_vals)],
                            color='#bbbbbb', lw=1.4, zorder=2)
                # Member dots (slight horizontal jitter to avoid coincident overlap)
                jit = rng.uniform(-0.025, 0.025, size=len(m_vals))
                ax.scatter(np.full(len(m_vals), x) + jit, m_vals,
                           s=58, c='#aaaaaa', edgecolors='white',
                           linewidths=0.5, zorder=3)
                # Ensemble star
                ax.scatter([x], [v_ens], s=200, marker='*',
                           c=self.COND_COLOR[cond],
                           edgecolors='black', linewidths=1.2, zorder=5)

        ax.set_xticks(range(n_ens))
        ax.set_xticklabels(self.ENSEMBLE_ORDER, fontsize=12.5,
                           rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, n_ens - 0.5)

        # Vertical separators between ensembles
        for k in range(1, n_ens):
            ax.axvline(k - 0.5, color='#eeeeee', lw=0.8, zorder=1)

    def _panel_a(self, ax):
        self._draw_ens_vs_member(ax, 'acc', 'Accuracy (%)', (40, 100),
                                 lower_better=False)
        self.add_panel_label(ax, 'a',
                             'Accuracy: ensembles land near the best member')

    def _panel_b(self, ax):
        self._draw_ens_vs_member(ax, 'high_risk',
                                 'High-risk error rate (%)',
                                 (-1, 18), lower_better=True)
        self.add_panel_label(ax, 'b',
                             'High-risk error: not lower than safest member')

    def _panel_c(self, ax):
        self._draw_ens_vs_member(ax, 'dang_oc',
                                 'Dangerous overconf. rate (%)',
                                 (-1, 35), lower_better=True)
        self.add_panel_label(ax, 'c',
                             'Dangerous overconf. persists in ensembles')

    # ----- Panel d: synchronized failure bars per ensemble × condition -----
    def _panel_d(self, ax):
        n_ens = len(self.ENSEMBLE_ORDER)
        n_cond = len(self.CONDITIONS)
        x = np.arange(n_ens)
        width = 0.27

        for i_c, cond in enumerate(self.CONDITIONS):
            vals = []
            for ens in self.ENSEMBLE_ORDER:
                row = self.df_ens[(self.df_ens['ensemble'] == ens) &
                                  (self.df_ens['Condition'] == cond)].iloc[0]
                vals.append(row['sync_fail'])
            xs = x + (i_c - (n_cond - 1) / 2) * width
            ax.bar(xs, vals, width=width, color=self.COND_COLOR[cond],
                   edgecolor='white', linewidth=0.7, alpha=0.92, zorder=3)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.18, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=11,
                        color=self.COND_COLOR[cond], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(self.ENSEMBLE_ORDER, fontsize=12.5,
                           rotation=15, ha='right')
        ax.set_ylabel('Synchronized failure rate (%)')
        ax.set_ylim(0, 12)
        for k in range(1, n_ens):
            ax.axvline(k - 0.5, color='#eeeeee', lw=0.8, zorder=1)

        self.add_panel_label(ax, 'd',
                             'Synchronized failure is a substantial fraction of errors')

    # ----- Panel e: synchronized failure vs high-risk error scatter -----
    def _panel_e(self, ax):
        # Diagonal reference: SF = HR (every high-risk error is synchronized)
        max_v = 12
        ax.plot([0, max_v], [0, max_v], color='black', lw=1.3, ls='--',
                zorder=1, label='SF = HR (all errors synchronized)')

        # Plot one point per (ensemble, condition); marker shape per ensemble,
        # color per condition
        markers = {'Dense Mid': 'o', 'Frontier': 's',
                   'Qwen scale': 'D', 'Cross scale': '^'}
        for cond in self.CONDITIONS:
            for ens in self.ENSEMBLE_ORDER:
                row = self.df_ens[(self.df_ens['ensemble'] == ens) &
                                  (self.df_ens['Condition'] == cond)].iloc[0]
                ax.scatter([row['high_risk']], [row['sync_fail']],
                           s=180, marker=markers[ens],
                           c=self.COND_COLOR[cond],
                           edgecolors='black', linewidths=1.0,
                           alpha=0.95, zorder=4)

        ax.set_xlabel('High-risk error rate (%)')
        ax.set_ylabel('Synchronized failure rate (%)')
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.set_aspect('equal', adjustable='box')

        # Inline annotation about the diagonal
        ax.text(0.04, 0.96,
                'Points near diagonal:\nensemble failures are coordinated',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11, style='italic', color='#222',
                bbox=dict(facecolor='white', edgecolor='#bbb',
                          alpha=0.92, pad=4, lw=0.5))

        # Marker (ensemble) legend at bottom right
        marker_handles = [
            Line2D([0], [0], marker=markers[e], mfc='#aaaaaa',
                   mec='black', ms=12, lw=0, label=e)
            for e in self.ENSEMBLE_ORDER
        ]
        ax.legend(handles=marker_handles, loc='lower right',
                  frameon=False, fontsize=11, handletextpad=0.4,
                  title='Ensemble', title_fontsize=11)

        self.add_panel_label(ax, 'e',
                             'Most ensemble high-risk errors are synchronized')

    # ----- Panel f: ensemble vs best member, summary deltas -----
    def _panel_f(self, ax):
        # For each metric, compute (ensemble - best_member) over all
        # 4 ensembles × 3 conditions = 12 cases
        metrics = [('acc', 'Δ Accuracy', '#2166AC', False),
                   ('high_risk', 'Δ High-risk', '#D6604D', True),
                   ('unsafe', 'Δ Unsafe', '#7F0F0F', True),
                   ('dang_oc', 'Δ Dang. overconf.', '#B2182B', True)]

        labels, means, sems, colors = [], [], [], []
        for key, lbl, col, lower_better in metrics:
            deltas = []
            for ens in self.ENSEMBLE_ORDER:
                for cond in self.CONDITIONS:
                    row = self.df_ens[
                        (self.df_ens['ensemble'] == ens) &
                        (self.df_ens['Condition'] == cond)].iloc[0]
                    v_ens = row[key]
                    m_vals = self._member_metric(ens, cond, key)
                    if not m_vals or pd.isna(v_ens):
                        continue
                    if lower_better:
                        # best member = lowest; improvement = best - ensemble
                        best = min(m_vals)
                        d = best - v_ens
                    else:
                        # best member = highest; improvement = ensemble - best
                        best = max(m_vals)
                        d = v_ens - best
                    deltas.append(d)
            arr = np.array(deltas, dtype=float)
            arr = arr[~np.isnan(arr)]
            labels.append(lbl)
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
            colors.append(col)

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, width=0.62,
               color=colors, edgecolor='white', linewidth=0.8,
               alpha=0.92, zorder=3,
               error_kw=dict(elinewidth=1.6, capsize=6, ecolor='#222'))
        for xi, m, sem in zip(x, means, sems):
            offset = sem + 0.18 if m >= 0 else -(sem + 0.18)
            va = 'bottom' if m >= 0 else 'top'
            ax.text(xi, m + offset, f'{m:+.2f}',
                    ha='center', va=va, fontsize=12.5, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12.5)
        ax.set_ylabel('Mean change vs.\nbest member (pp)')
        ax.set_ylim(-4, 1)
        self.add_panel_label(ax, 'f',
                             'Ensembles do not beat the best member on safety')

    # ----- Panel g: dang_oc per ensemble × condition -----
    def _panel_g(self, ax):
        n_ens = len(self.ENSEMBLE_ORDER)
        n_cond = len(self.CONDITIONS)
        x = np.arange(n_ens)
        width = 0.27

        for i_c, cond in enumerate(self.CONDITIONS):
            vals = []
            for ens in self.ENSEMBLE_ORDER:
                row = self.df_ens[(self.df_ens['ensemble'] == ens) &
                                  (self.df_ens['Condition'] == cond)].iloc[0]
                vals.append(row['dang_oc'])
            xs = x + (i_c - (n_cond - 1) / 2) * width
            ax.bar(xs, vals, width=width, color=self.COND_COLOR[cond],
                   edgecolor='white', linewidth=0.7, alpha=0.92, zorder=3,
                   label=self.COND_DISPLAY[cond])
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.4, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=11,
                        color=self.COND_COLOR[cond], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(self.ENSEMBLE_ORDER, fontsize=12.5,
                           rotation=15, ha='right')
        ax.set_ylabel('Dangerous overconf. rate (%)')
        ax.set_ylim(0, 22)
        for k in range(1, n_ens):
            ax.axvline(k - 0.5, color='#eeeeee', lw=0.8, zorder=1)

        self.add_panel_label(ax, 'g',
                             'Condition still dominates ensemble structure')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')
        print(f'Saved: {self.OUT_PNG}')


if __name__ == '__main__':
    Figure6().build().save()
