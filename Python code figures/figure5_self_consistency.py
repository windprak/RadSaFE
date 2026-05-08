"""
figure5_self_consistency.py
Figure 5: Inference-time compute (self-consistency) does not produce safety.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Figure5:
    SC_CSV = './self_consistency.csv'
    OUT_PNG = './figs/figure5_self_consistency.png'

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

    def __init__(self):
        self._set_rcparams()
        self.df = self._load()
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

    @staticmethod
    def _short_name(m):
        # Strip path prefix and substitution annotation
        s = re.sub(r'\s*\[substituted for [^\]]+\]', '', m)
        s = s.split('/')[-1]
        s = s.replace('-Instruct', '').replace('-it', '')
        s = s.replace('Meta-', '')
        return s.strip()

    def _load(self):
        df = pd.read_csv(self.SC_CSV)
        df['acc'] = df['Accuracy (mean ± std [95% CI])'].apply(
            lambda s: float(str(s).split('±')[0].strip()))
        df = df.rename(columns={
            'High risk error (rate)': 'high_risk',
            'Unsafe answer (rate)': 'unsafe',
            'Contradiction (rate)': 'contradiction',
            'Dangerous overconfidence (rate)': 'dang_oc',
            'mean confidence': 'mean_conf_str',
            'Mean latency (seconds)': 'latency',
            'Compute regime': 'regime',
        })
        df['short'] = df['Model'].apply(self._short_name)
        # Parse mean_confidence string "96.0 ± 11.0" → float
        def parse_conf(s):
            if pd.isna(s) or str(s).strip() == '':
                return np.nan
            return float(str(s).split('±')[0].strip())
        df['mean_conf'] = df['mean_conf_str'].apply(parse_conf)
        return df

    @staticmethod
    def add_panel_label(ax, letter, title='', x=-0.18, y=1.04):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='bottom', ha='left')
        if title:
            ax.text(x + 0.08, y, title, transform=ax.transAxes,
                    fontsize=14, fontweight='normal', va='bottom', ha='left')

    def build(self):
        self.fig = plt.figure(figsize=(17, 21))

        # Top legend strip - increased distance from row below
        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.998, bottom=0.960)
        # Row 1: panels a, b, c — three slopegraphs
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.07, right=0.98,
                                      top=0.938, bottom=0.690, wspace=0.34)
        # Row 2: panels d, e - increased distance from row 3
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.642, bottom=0.380,
                                      wspace=0.32, width_ratios=[1, 1.4])
        # Row 3: panels f, g - increased distance from row 2
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.308, bottom=0.045,
                                      wspace=0.32, width_ratios=[1, 1.05])

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
        items = [
            Line2D([0], [0], marker='o', color=self.COND_COLOR[c],
                   mfc=self.COND_COLOR[c], mec='white', ms=12, lw=2.4,
                   label=self.COND_DISPLAY[c])
            for c in self.CONDITIONS
        ]
        ax.legend(handles=items, loc='center', ncol=3, frameon=False,
                  fontsize=14, handletextpad=0.5, columnspacing=2.5,
                  title='Condition', title_fontsize=14)

    # ----- Slopegraph helper -----
    def _draw_slopegraph(self, ax, metric, ylabel, ylim, lower_better,
                         show_xlabels=True):
        """Single→SC paired transitions for each (model, condition).
        x = {0: Single, 1: SC}; lines per pair, color by condition."""
        cond_means = []  # collect (cond, m_s, m_c) for end-label placement

        for cond in self.CONDITIONS:
            col = self.COND_COLOR[cond]
            sub = self.df[self.df['Condition'] == cond]
            for short in sub['short'].unique():
                row_s = sub[(sub['short'] == short) &
                            (sub['regime'] == 'Single')]
                row_c = sub[(sub['short'] == short) &
                            (sub['regime'] == 'Self-consistency')]
                if not (len(row_s) and len(row_c)):
                    continue
                v0 = row_s[metric].values[0]
                v1 = row_c[metric].values[0]
                if pd.isna(v0) or pd.isna(v1):
                    continue
                # darker if change worsens
                worsens = (v1 > v0) if lower_better else (v1 < v0)
                lw = 2.2 if worsens else 1.4
                alpha = 0.95 if worsens else 0.7
                ax.plot([0, 1], [v0, v1], '-', color=col, lw=lw,
                        alpha=alpha, zorder=3)
                ax.scatter([0], [v0], s=70, c=col, edgecolors='white',
                           linewidths=0.7, zorder=4)
                ax.scatter([1], [v1], s=70, c=col, edgecolors='white',
                           linewidths=0.7, zorder=4)

            # Compute condition-mean slope (thick black)
            pairs = []
            for short in sub['short'].unique():
                row_s = sub[(sub['short'] == short) &
                            (sub['regime'] == 'Single')]
                row_c = sub[(sub['short'] == short) &
                            (sub['regime'] == 'Self-consistency')]
                if (len(row_s) and len(row_c) and
                        not pd.isna(row_s[metric].values[0]) and
                        not pd.isna(row_c[metric].values[0])):
                    pairs.append((row_s[metric].values[0],
                                  row_c[metric].values[0]))
            if pairs:
                m_s = np.mean([p[0] for p in pairs])
                m_c = np.mean([p[1] for p in pairs])
                ax.plot([0, 1], [m_s, m_c], '-', color=col,
                        lw=4.0, alpha=1.0, zorder=5)
                ax.scatter([0, 1], [m_s, m_c], s=200, c=col,
                           edgecolors='black', linewidths=1.6, zorder=6)
                cond_means.append((cond, col, m_s, m_c))

        # Place Δ labels with vertical de-overlap
        # sort by m_c (right endpoint) and offset crowded labels
        cond_means.sort(key=lambda r: r[3])
        y_range = ylim[1] - ylim[0]
        min_gap = 0.06 * y_range
        last_y = -np.inf
        for cond, col, m_s, m_c in cond_means:
            y_text = max(m_c, last_y + min_gap)
            delta = m_c - m_s
            ax.annotate(f'Δ = {delta:+.1f}',
                        xy=(1.0, m_c), xytext=(1.07, y_text),
                        textcoords='data',
                        color=col, fontweight='bold', fontsize=11.5,
                        va='center', ha='left',
                        arrowprops=dict(arrowstyle='-', color=col,
                                        lw=0.8, alpha=0.6))
            last_y = y_text

        ax.set_xticks([0, 1])
        if show_xlabels:
            ax.set_xticklabels(['Single', 'Self-\nconsistency'], fontsize=13)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-0.25, 1.55)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    def _panel_a(self, ax):
        self._draw_slopegraph(
            ax, 'acc', 'Accuracy (%)', (50, 100), lower_better=False)
        self.add_panel_label(ax, 'a',
                             'Accuracy: small upward shift under self-consistency')

    def _panel_b(self, ax):
        self._draw_slopegraph(
            ax, 'high_risk', 'High-risk error rate (%)', (-1, 25),
            lower_better=True)
        self.add_panel_label(ax, 'b',
                             'High-risk error: largely unchanged')

    def _panel_c(self, ax):
        self._draw_slopegraph(
            ax, 'unsafe', 'Unsafe answer rate (%)', (0, 5),
            lower_better=True)
        self.add_panel_label(ax, 'c',
                             'Unsafe answers: largely unchanged')

    # ----- Panel d: Δacc vs Δhigh_risk_safety per (model, condition) -----
    def _panel_d(self, ax):
        for cond in self.CONDITIONS:
            sub = self.df[self.df['Condition'] == cond]
            xs, ys = [], []
            for short in sub['short'].unique():
                row_s = sub[(sub['short'] == short) &
                            (sub['regime'] == 'Single')]
                row_c = sub[(sub['short'] == short) &
                            (sub['regime'] == 'Self-consistency')]
                if not (len(row_s) and len(row_c)):
                    continue
                d_acc = row_c['acc'].values[0] - row_s['acc'].values[0]
                # y axis: positive = SC reduced high-risk = safer
                d_hr_safety = (row_s['high_risk'].values[0] -
                               row_c['high_risk'].values[0])
                xs.append(d_acc)
                ys.append(d_hr_safety)
            ax.scatter(xs, ys, s=110, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.8,
                       alpha=0.92, zorder=4)

        ax.axhline(0, color='black', lw=0.8, zorder=1)
        ax.axvline(0, color='black', lw=0.8, zorder=1)

        # Quadrant labels
        ax.text(0.97, 0.97, 'better accuracy\nbetter safety',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, color='#3c6e2c', style='italic',
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=2))
        ax.text(0.03, 0.97, 'worse accuracy\nbetter safety',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11, color='#555', style='italic',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=2))
        ax.text(0.97, 0.03, 'better accuracy\nworse safety',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, color='#a04444', style='italic',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=2))
        ax.text(0.03, 0.03, 'worse accuracy\nworse safety',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=11, color='#555', style='italic',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=2))

        ax.set_xlabel('Δ Accuracy (SC − Single, pp)')
        ax.set_ylabel('Δ High-risk safety (Single − SC, pp)')
        ax.set_xlim(-3, 5)
        ax.set_ylim(-2, 2)
        self.add_panel_label(ax, 'd',
                             'Accuracy and safety changes are decoupled')

    # ----- Panel e: dang_oc per (model, condition) under self-consistency -----
    def _panel_e(self, ax):
        sc = self.df[self.df['regime'] == 'Self-consistency'].copy()
        models_short = sc['short'].unique().tolist()
        # Sort models by overall mean dang_oc
        order = (sc.groupby('short')['dang_oc'].mean()
                   .sort_values().index.tolist())

        x = np.arange(len(order))
        width = 0.27
        n_cond = len(self.CONDITIONS)

        for i_c, cond in enumerate(self.CONDITIONS):
            sub_c = sc[sc['Condition'] == cond].set_index('short')
            vals = [sub_c['dang_oc'].get(m, np.nan) for m in order]
            xs = x + (i_c - (n_cond - 1) / 2) * width
            bars = ax.bar(xs, vals, width=width,
                          color=self.COND_COLOR[cond],
                          edgecolor='white', linewidth=0.7,
                          alpha=0.92, zorder=3)
            # value labels
            for xi, v in zip(xs, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.6, f'{v:.0f}',
                            ha='center', va='bottom',
                            fontsize=10.5, color=self.COND_COLOR[cond],
                            fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha='right', fontsize=11.5)
        ax.set_ylabel('Dangerous overconf. rate under SC (%)')
        ax.set_ylim(0, 45)

        self.add_panel_label(ax, 'e',
                             'SC produces high dangerous overconf. across models')

    # ----- Panel f: SC mean confidence vs SC high-risk per (model, cond) -----
    def _panel_f(self, ax):
        sc = self.df[self.df['regime'] == 'Self-consistency'].copy()
        for cond in self.CONDITIONS:
            sub = sc[sc['Condition'] == cond]
            ax.scatter(sub['high_risk'], sub['mean_conf'],
                       s=110, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.8,
                       alpha=0.92, zorder=4,
                       label=self.COND_DISPLAY[cond])

        # If confidence were a useful safety signal, points with high HR would have low confidence
        # → expect a downward slope. Plot a zero-discrimination reference:
        # a horizontal line at the mean confidence
        mean_c = sc['mean_conf'].mean()
        ax.axhline(mean_c, color='#777', lw=1.0, ls=':', zorder=2)
        ax.text(0.97, mean_c + 0.6,
                f'overall mean confidence = {mean_c:.1f}%',
                fontsize=11, color='#444', ha='right', va='bottom',
                style='italic',
                transform=ax.get_yaxis_transform())

        ax.set_xlabel('High-risk error rate under SC (%)')
        ax.set_ylabel('Mean SC confidence (%)')
        ax.set_xlim(-0.5, 24)
        ax.set_ylim(80, 102)
        self.add_panel_label(ax, 'f',
                             'SC confidence is uniformly high regardless of safety')

    # ----- Panel g: summary delta bars (across all 8 models × 3 conds) -----
    def _panel_g(self, ax):
        metrics = [('acc', 'Δ Accuracy', '#2166AC', False),
                   ('high_risk', 'Δ High-risk', '#D6604D', True),
                   ('unsafe', 'Δ Unsafe', '#7F0F0F', True),
                   ('contradiction', 'Δ Contradiction', '#F4A582', True)]

        labels, means, sems, colors = [], [], [], []
        for key, lbl, col, lower_better in metrics:
            deltas = []
            for cond in self.CONDITIONS:
                sub = self.df[self.df['Condition'] == cond]
                for short in sub['short'].unique():
                    row_s = sub[(sub['short'] == short) &
                                (sub['regime'] == 'Single')]
                    row_c = sub[(sub['short'] == short) &
                                (sub['regime'] == 'Self-consistency')]
                    if not (len(row_s) and len(row_c)):
                        continue
                    d = row_c[key].values[0] - row_s[key].values[0]
                    if lower_better:
                        d = -d  # flip so positive = improvement
                    deltas.append(d)
            arr = np.array(deltas, dtype=float)
            arr = arr[~np.isnan(arr)]
            labels.append(lbl)
            means.append(arr.mean())
            sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
            colors.append(col)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=sems, width=0.62,
                      color=colors, edgecolor='white', linewidth=0.8,
                      alpha=0.92, zorder=3,
                      error_kw=dict(elinewidth=1.6, capsize=6, ecolor='#222'))
        
        # Annotate values, positioning above error bar cap
        for xi, m, sem in zip(x, means, sems):
            # Place text above the upper error bar cap
            y_pos = m + sem + 0.08
            ax.text(xi, y_pos, f'{m:+.2f}',
                    ha='center', va='bottom', fontsize=12.5, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12.5)
        ax.set_ylabel('Mean change (pp)')
        ax.set_ylim(0, 0.8)
        
        self.add_panel_label(ax, 'g',
                             'Self-consistency: mean shift across all metrics')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')
        print(f'Saved: {self.OUT_PNG}')


if __name__ == '__main__':
    Figure5().build().save()
