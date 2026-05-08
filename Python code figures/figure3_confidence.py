"""
figure3_confidence.py
Figure 3: Confidence is not a safety signal.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


class Figure3:
    MAIN_CSV = './Main_results.csv'
    OUT_PNG = './figs/figure3_confidence.png'

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

    OUTCOME_COLOR = {
        'correct':    '#2C7A2C',
        'incorrect':  '#9C9C9C',
        'high-risk':  '#D6604D',
        'unsafe':     '#7F0F0F',
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
            'Mean confidence': 'conf_all',
            'Mean confidence correct': 'conf_correct',
            'Mean confidence incorrect': 'conf_incorrect',
            'Mean confidence high risk errors': 'conf_hr',
            'Mean confidence unsafe erros': 'conf_unsafe',
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
        self.fig = plt.figure(figsize=(17, 23))

        # Top legend strip
        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.998, bottom=0.974)
        # Row 1: panel a (wide, 2 cols), panel b (narrower)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.953, bottom=0.760,
                                      wspace=0.30, width_ratios=[1.55, 1])
        # Row 2: panels c, d (two scatters)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.715, bottom=0.522, wspace=0.30)
        # Row 3: panels e, f (paired)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.477, bottom=0.284, wspace=0.30)
        # Row 4: panel g (full-width heatmap)
        gs_r4 = self.fig.add_gridspec(1, 1, left=0.07, right=0.98,
                                      top=0.239, bottom=0.043)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_e(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_f(self.fig.add_subplot(gs_r3[0, 1]))
        self._panel_g(self.fig.add_subplot(gs_r4[0, 0]))
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

    # ----- Panel a: confidence by outcome class, faceted by 4 conditions -----
    def _panel_a(self, ax):
        # Subset of conditions to keep visualization legible
        conds_show = ['Closed-book', 'Clean evidence',
                      'Standard RAG', 'RaR']
        outcome_keys = ['conf_correct', 'conf_incorrect', 'conf_hr',
                        'conf_unsafe']
        outcome_labels = ['Correct', 'Incorrect', 'High-risk', 'Unsafe']
        outcome_colors = [self.OUTCOME_COLOR['correct'],
                          self.OUTCOME_COLOR['incorrect'],
                          self.OUTCOME_COLOR['high-risk'],
                          self.OUTCOME_COLOR['unsafe']]

        n_cond = len(conds_show)
        n_out = len(outcome_keys)
        rng = np.random.default_rng(7)
        group_width = 0.85
        within = group_width / n_out

        for i_c, cond in enumerate(conds_show):
            sub = self.df_ind[self.df_ind['Condition'] == cond]
            for i_o, (key, lbl, col) in enumerate(
                    zip(outcome_keys, outcome_labels, outcome_colors)):
                vals = sub[key].dropna().values.astype(float)
                center = i_c + (i_o - (n_out - 1) / 2) * within
                # box
                if len(vals) > 0:
                    bp = ax.boxplot([vals], positions=[center],
                                    widths=within * 0.78,
                                    patch_artist=True, showfliers=False,
                                    medianprops=dict(color='black', lw=1.5),
                                    boxprops=dict(facecolor=col, alpha=0.35,
                                                  edgecolor=col, lw=1.0),
                                    whiskerprops=dict(color=col, lw=1.0),
                                    capprops=dict(color=col, lw=1.0))
                    # strip
                    jit = rng.uniform(-within * 0.22, within * 0.22, size=len(vals))
                    ax.scatter(center + jit, vals, s=18, c=col,
                               edgecolors='white', linewidths=0.4,
                               alpha=0.85, zorder=3)

        ax.set_xticks(np.arange(n_cond))
        ax.set_xticklabels([self.COND_DISPLAY[c] for c in conds_show],
                           fontsize=14)
        ax.set_ylabel('Mean confidence (%)')
        ax.set_ylim(35, 102)

        # Outcome legend placed inside, lower-left corner, with white bg
        legend_handles = [
            Line2D([0], [0], marker='s', color=col, mfc=col, mec='white',
                   ms=12, lw=0, label=lbl)
            for col, lbl in zip(outcome_colors, outcome_labels)
        ]
        ax.legend(handles=legend_handles, loc='lower left',
                  ncol=4, frameon=True, fontsize=12, handletextpad=0.3,
                  columnspacing=1.0, title='Outcome class',
                  title_fontsize=12,
                  facecolor='white', edgecolor='#cccccc',
                  framealpha=0.95)

        # Vertical dividers between condition groups
        for k in range(1, n_cond):
            ax.axvline(k - 0.5, color='#dddddd', lw=0.8, zorder=0)

        self.add_panel_label(ax, 'a',
                             'Confidence distributions overlap across outcome classes')

    # ----- Panel b: separation gap conf_correct - conf_high_risk per model -----
    def _panel_b(self, ax):
        rng = np.random.default_rng(11)
        x_pos = np.arange(len(self.CONDITIONS))
        for i, cond in enumerate(self.CONDITIONS):
            sub = self.df_ind[self.df_ind['Condition'] == cond]
            gaps = (sub['conf_correct'] - sub['conf_hr']).dropna().values
            jit = rng.uniform(-0.25, 0.25, size=len(gaps))
            ax.scatter(np.full(len(gaps), i) + jit, gaps,
                       s=46, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.6,
                       alpha=0.78, zorder=3)
            med = np.median(gaps)
            ax.plot([i - 0.32, i + 0.32], [med, med],
                    color='black', lw=2.6, zorder=4)
            ax.text(i + 0.36, med, f'{med:+.1f}',
                    va='center', ha='left', fontsize=12,
                    fontweight='bold')

        # Reference: a "useful filter" gap
        ax.axhline(30, color='#3c6e2c', lw=1.4, ls='--', zorder=1)
        ax.text(len(self.CONDITIONS) - 0.5, 32,
                'useful safety filter (gap $\\geq$ 30 pp)',
                fontsize=11.5, color='#3c6e2c', ha='right', va='bottom',
                style='italic')
        ax.axhline(0, color='black', lw=0.8, zorder=1)

        ax.set_xticks(x_pos)
        # short labels with line break to fit
        short = {'Closed-book': 'Closed-\nbook',
                 'Clean evidence': 'Clean\nevidence',
                 'Conflict evidence': 'Conflict\nevidence',
                 'Standard RAG': 'Standard\nRAG',
                 'RaR': 'Agentic\nRAG',
                 'Max context': 'Max\ncontext'}
        ax.set_xticklabels([short[c] for c in self.CONDITIONS],
                           ha='center', fontsize=12.5)
        ax.set_ylabel('Confidence gap (correct $-$ high-risk, pp)')
        ax.set_ylim(-25, 60)
        ax.set_xlim(-0.6, len(self.CONDITIONS) - 0.4)

        self.add_panel_label(ax, 'b',
                             'Confidence gap is too small to filter unsafe outputs')

    # ----- Panel c: conf_incorrect vs accuracy per (model, condition) -----
    def _panel_c(self, ax):
        for cond in self.CONDITIONS:
            sub = self.df_ind[self.df_ind['Condition'] == cond]
            ax.scatter(sub['acc'], sub['conf_incorrect'],
                       s=44, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.6,
                       alpha=0.78, zorder=3)

        # Single overall trend across all (model, condition) points
        all_x = self.df_ind['acc'].values
        all_y = self.df_ind['conf_incorrect'].values
        m = ~(np.isnan(all_x) | np.isnan(all_y))
        if m.sum() >= 2:
            coef = np.polyfit(all_x[m], all_y[m], 1)
            xs = np.linspace(all_x[m].min(), all_x[m].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), color='black', lw=1.8,
                    ls='--', alpha=0.7, zorder=2,
                    label=f'overall slope = {coef[0]:+.2f} pp/pp')
            ax.legend(loc='lower right', frameon=False, fontsize=11.5)

        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('Confidence among incorrect answers (%)')
        ax.set_xlim(15, 100)
        ax.set_ylim(40, 102)
        self.add_panel_label(ax, 'c',
                             'Higher accuracy does not humble residual errors')

    # ----- Panel d: conf_hr vs conf_correct per (model, condition) -----
    def _panel_d(self, ax):
        for cond in self.CONDITIONS:
            sub = self.df_ind[self.df_ind['Condition'] == cond]
            ax.scatter(sub['conf_correct'], sub['conf_hr'],
                       s=44, c=self.COND_COLOR[cond],
                       edgecolors='white', linewidths=0.6,
                       alpha=0.78, zorder=3)
        # diagonal y = x
        lim = [40, 102]
        ax.plot(lim, lim, color='black', lw=1.2, ls='--', zorder=1)
        ax.text(55, 58, 'y = x', fontsize=12.5, ha='left',
                color='black', style='italic',
                rotation=45, rotation_mode='anchor')

        # shade region where conf_hr is meaningfully below conf_correct
        ax.fill_between([40, 102], [40 - 30, 102 - 30], [40, 102],
                        color='#e8f0e0', alpha=0.45, zorder=0)
        ax.text(95, 50,
                'discriminative region\n(gap $\\geq$ 30 pp)',
                fontsize=11.5, color='#3c6e2c', ha='right', va='center',
                style='italic',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=2))

        ax.set_xlabel('Confidence among correct (%)')
        ax.set_ylabel('Confidence among high-risk errors (%)')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal', adjustable='box')
        self.add_panel_label(ax, 'd',
                             'Confidence in dangerous errors tracks confidence in correct answers')

    # ----- Panels e, f: paired dangerous overconfidence transitions -----
    def _draw_paired_doc(self, ax, cond_from, cond_to,
                         color_from, color_to,
                         label_from, label_to):
        df_from = self.df_ind[self.df_ind['Condition'] == cond_from
                              ].set_index('Model')
        df_to = self.df_ind[self.df_ind['Condition'] == cond_to
                            ].set_index('Model')
        common = sorted(df_from.index.intersection(df_to.index),
                        key=lambda m: df_from.loc[m, 'dang_oc'])
        n = len(common)
        y = np.arange(n)
        for i, m in enumerate(common):
            v0 = df_from.loc[m, 'dang_oc']
            v1 = df_to.loc[m, 'dang_oc']
            # color the connector by direction (improvement vs worsening)
            cline = '#888888' if v1 <= v0 else '#cc4444'
            ax.plot([v0, v1], [i, i], color=cline, lw=1.0, alpha=0.7,
                    zorder=2)
        # endpoint dots
        v_from = [df_from.loc[m, 'dang_oc'] for m in common]
        v_to = [df_to.loc[m, 'dang_oc'] for m in common]
        ax.scatter(v_from, y, s=58, c=color_from, edgecolors='white',
                   linewidths=0.7, zorder=3, label=label_from)
        ax.scatter(v_to, y, s=58, c=color_to, edgecolors='white',
                   linewidths=0.7, zorder=3, label=label_to)

        # Median markers
        med_from = np.median(v_from)
        med_to = np.median(v_to)
        ax.axvline(med_from, color=color_from, lw=1.5, ls=':',
                   alpha=0.85, zorder=1)
        ax.axvline(med_to, color=color_to, lw=1.5, ls=':',
                   alpha=0.85, zorder=1)

        # Worsening count
        worse = sum(1 for v0, v1 in zip(v_from, v_to) if v1 > v0)
        better = sum(1 for v0, v1 in zip(v_from, v_to) if v1 < v0)
        return n, worse, better, med_from, med_to

    def _panel_e(self, ax):
        n, worse, better, m_from, m_to = self._draw_paired_doc(
            ax, 'Closed-book', 'Clean evidence',
            self.COND_COLOR['Closed-book'],
            self.COND_COLOR['Clean evidence'],
            'Closed-book', 'Clean evidence')

        ax.set_xlabel('Dangerous overconfidence rate (%)')
        ax.set_ylabel('Models (sorted by closed-book value)')
        ax.set_yticks([])
        ax.set_xlim(-2, 50)
        # Summary box at bottom-right
        ax.text(0.97, 0.03,
                'closed-book $\\rightarrow$ clean evidence\n\n'
                f'{better}/{n} models improved\n{worse}/{n} worsened\n'
                f'median: {m_from:.1f} $\\rightarrow$ {m_to:.1f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12,
                bbox=dict(facecolor='white', edgecolor='#888',
                          alpha=0.92, pad=5, lw=0.6))

        self.add_panel_label(ax, 'e',
                             'Clean evidence: universal collapse of dangerous overconfidence')

    def _panel_f(self, ax):
        n, worse, better, m_from, m_to = self._draw_paired_doc(
            ax, 'Closed-book', 'RaR',
            self.COND_COLOR['Closed-book'],
            self.COND_COLOR['RaR'],
            'Closed-book', 'Agentic RAG')

        ax.set_xlabel('Dangerous overconfidence rate (%)')
        ax.set_ylabel('Models (sorted by closed-book value)')
        ax.set_yticks([])
        ax.set_xlim(-2, 50)

        ax.text(0.97, 0.03,
                'closed-book $\\rightarrow$ agentic RAG\n\n'
                f'{better}/{n} models improved\n{worse}/{n} worsened\n'
                f'median: {m_from:.1f} $\\rightarrow$ {m_to:.1f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12,
                bbox=dict(facecolor='white', edgecolor='#888',
                          alpha=0.92, pad=5, lw=0.6))

        self.add_panel_label(ax, 'f',
                             'Agentic RAG: many models stay or worsen')

    # ----- Panel g: heatmap dang_oc per (model, condition) -----
    def _panel_g(self, ax):
        # Sort models by closed-book dang_oc, ascending
        cb = self.df_ind[self.df_ind['Condition'] == 'Closed-book'
                         ].set_index('Model')['dang_oc']
        models_sorted = cb.sort_values().index.tolist()

        mat = np.full((len(models_sorted), len(self.CONDITIONS)), np.nan)
        for i, m in enumerate(models_sorted):
            for j, cond in enumerate(self.CONDITIONS):
                row = self.df_ind[(self.df_ind['Model'] == m) &
                                  (self.df_ind['Condition'] == cond)]
                if len(row):
                    mat[i, j] = row['dang_oc'].values[0]

        im = ax.imshow(mat.T, aspect='auto', cmap='Reds', vmin=0, vmax=50,
                       interpolation='nearest')
        # Annotate cells with value - increased font size
        for j in range(mat.shape[1]):
            for i in range(mat.shape[0]):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = 'white' if v > 25 else '#222'
                    ax.text(i, j, f'{v:.0f}', ha='center', va='center',
                            fontsize=10.5, color=txt_color)

        # short model labels
        def short_name(m):
            s = m.split('/')[-1]
            s = s.replace('-Instruct', '').replace('-it', '')
            return s

        ax.set_xticks(range(len(models_sorted)))
        ax.set_xticklabels([short_name(m) for m in models_sorted],
                           rotation=55, ha='right', fontsize=10)
        ax.set_yticks(range(len(self.CONDITIONS)))
        ax.set_yticklabels([self.COND_DISPLAY[c] for c in self.CONDITIONS],
                           fontsize=13)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        for s in ax.spines.values():
            s.set_visible(False)

        # Colorbar
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012,
                                 aspect=12)
        cbar.set_label('Dangerous overconf. rate (%)', fontsize=12.5)
        cbar.ax.tick_params(labelsize=11)

        # Align panel label with panel e's x position
        self.add_panel_label(ax, 'g',
                             'Dangerous overconfidence is governed by condition, not by model',
                             x=-0.09, y=1.04)

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')
        print(f'Saved: {self.OUT_PNG}')


if __name__ == '__main__':
    Figure3().build().save()
