"""
figure4_scaling.py
Figure 4: Safety and accuracy follow different scaling laws.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Figure4:
    MAIN_CSV = './Main_results.csv'
    OUT_PNG = './figs/figure4_scaling.png'

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

    PARAMS = {
        'Qwen/Qwen2.5-0.5B-Instruct': 0.5, 'Qwen/Qwen2.5-1.5B-Instruct': 1.5,
        'Qwen/Qwen2.5-3B-Instruct': 3, 'Qwen/Qwen2.5-7B-Instruct': 7,
        'Qwen/Qwen2.5-14B-Instruct': 14, 'Qwen/Qwen2.5-32B-Instruct': 32,
        'Qwen/Qwen3-4B': 4, 'Qwen/Qwen3-8B': 8,
        'Qwen/Qwen3-14B': 14, 'Qwen/Qwen3-32B': 32,
        'Qwen/Qwen3-VL-235B-A22B-Instruct': 235,
        'deepseek-ai/DeepSeek-R1': 671, 'deepseek-ai/DeepSeek-V3.2': 685,
        'google/gemma-3-4b-it': 4, 'google/gemma-3-12b-it': 12,
        'google/gemma-3-27b-it': 27, 'google/gemma-4-31B-it': 31,
        'google/gemma-4-E4B-it': 4,
        'google/medgemma-1.5-4b-it': 4, 'google/medgemma-27b-text-it': 27,
        'gpt-oss-20b': 20, 'gpt-oss-120b': 120,
        'meta-llama/Llama-3.2-1B-Instruct': 1,
        'meta-llama/Llama-3.2-3B-Instruct': 3,
        'meta-llama/Meta-Llama-3-8B-Instruct': 8,
        'meta-llama/Meta-Llama-3-70B-Instruct': 70,
        'meta-llama/Llama-3.3-70B-Instruct': 70,
        'meta-llama/Llama-4-Scout-17B-16E-Instruct': 17,
        'mistralai/Ministral-3-3B-Instruct-2512': 3,
        'mistralai/Ministral-3-8B-Instruct-2512': 8,
        'mistralai/Ministral-3-14B-Instruct-2512': 14,
        'mistralai/Mistral-Small-3.2-24B-Instruct-2506': 24,
        'mistralai/Mistral-Small-4-119B-2603': 119,
        'mistralai/Mistral-Large-3-675B-Instruct-2512': 675,
    }

    FAMILY_ORDER = ['Qwen', 'Llama', 'Gemma', 'MedGemma',
                    'DeepSeek', 'Mistral', 'OpenAI-OSS']

    FAMILY_COLOR = {
        'Qwen':       '#984EA3',
        'Llama':      '#E41A1C',
        'Gemma':      '#4DAF4A',
        'MedGemma':   '#7FBC41',
        'DeepSeek':   '#377EB8',
        'Mistral':    '#FF7F00',
        'OpenAI-OSS': '#A65628',
    }

    # Use the same condition palette as previous figures (for consistency
    # with heatmap and panel g)
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

    @staticmethod
    def _family_of(m):
        if 'medgemma' in m.lower(): return 'MedGemma'
        if 'gemma' in m.lower(): return 'Gemma'
        if 'qwen' in m.lower(): return 'Qwen'
        if 'llama' in m.lower(): return 'Llama'
        if 'deepseek' in m.lower(): return 'DeepSeek'
        if 'mistral' in m.lower() or 'ministral' in m.lower(): return 'Mistral'
        if 'gpt-oss' in m.lower(): return 'OpenAI-OSS'
        return 'Other'

    def _load(self):
        df = pd.read_csv(self.MAIN_CSV)
        df['acc'] = df['Accuracy (mean ± std [95% CI])'].apply(
            lambda s: float(str(s).split('±')[0].strip()))
        df = df.rename(columns={
            'High risk error (rate)': 'high_risk',
            'Unsafe answer (rate)': 'unsafe',
            'Contradiction (rate)': 'contradiction',
            'Dangerous overconfidence (rate)': 'dang_oc',
        })
        # Add params and family
        df['params'] = df['Model'].map(self.PARAMS)
        df['family'] = df['Model'].apply(self._family_of)
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

        # Top legend strip (families + closed-book vs clean markers)
        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.998, bottom=0.967)
        # Row 1: panels a, b, c — three scaling panels side by side
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.07, right=0.98,
                                      top=0.948, bottom=0.730, wspace=0.30)
        # Row 2: panel d (variance decomposition, narrow), panel e (heatmap, wide)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.683, bottom=0.420,
                                      wspace=0.35, width_ratios=[1, 1.6])
        # Row 3: panels f, g
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.07, right=0.98,
                                      top=0.350, bottom=0.045,
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
        # Family swatches
        family_items = [
            Line2D([0], [0], marker='o', color=self.FAMILY_COLOR[f],
                   mfc=self.FAMILY_COLOR[f], mec='white', ms=11, lw=0,
                   label=f)
            for f in self.FAMILY_ORDER
        ]
        # Marker style for condition (closed-book vs clean evidence)
        cond_items = [
            Line2D([0], [0], marker='o', mfc='#666', mec='#666', ms=10, lw=0,
                   label='Closed-book'),
            Line2D([0], [0], marker='*', mfc='white', mec='#222', ms=14,
                   mew=1.4, lw=0, label='Clean evidence'),
        ]
        leg1 = ax.legend(handles=family_items, loc='center left',
                         bbox_to_anchor=(0.0, 0.5),
                         ncol=7, frameon=False, fontsize=13.5,
                         handletextpad=0.4, columnspacing=1.2,
                         title='Model family', title_fontsize=13.5)
        ax.add_artist(leg1)
        ax.legend(handles=cond_items, loc='center right',
                  bbox_to_anchor=(1.0, 0.5),
                  ncol=2, frameon=False, fontsize=13.5,
                  handletextpad=0.5, columnspacing=1.2,
                  title='Marker', title_fontsize=13.5)

    # ----- Helper: per-(model) draw of within-family connectors -----
    def _scatter_family_scaling(self, ax, metric, ylim, ylabel, lower_better):
        """Plot one metric vs params, lines per family, two markers per model
        (closed-book filled circle, clean-evidence open star). Adds horizontal
        ceiling/floor band for clean-evidence model-average."""
        # Ceiling/floor band for clean evidence
        clean_avg = self.df_avg[
            self.df_avg['Condition'] == 'Clean evidence'].iloc[0]
        cb_avg = self.df_avg[
            self.df_avg['Condition'] == 'Closed-book'].iloc[0]
        v_clean = clean_avg[metric]
        v_cb = cb_avg[metric]

        for fam in self.FAMILY_ORDER:
            sub = self.df_ind[self.df_ind['family'] == fam].copy()
            sub_cb = sub[sub['Condition'] == 'Closed-book'
                         ].sort_values('params')
            sub_cl = sub[sub['Condition'] == 'Clean evidence'
                         ].sort_values('params')

            col = self.FAMILY_COLOR[fam]

            # Closed-book trend line for this family
            if len(sub_cb) >= 2:
                ax.plot(sub_cb['params'], sub_cb[metric], '-',
                        color=col, lw=1.6, alpha=0.55, zorder=2)
            # Clean-evidence trend line
            if len(sub_cl) >= 2:
                ax.plot(sub_cl['params'], sub_cl[metric], '--',
                        color=col, lw=1.4, alpha=0.55, zorder=2)

            # Markers
            ax.scatter(sub_cb['params'], sub_cb[metric],
                       s=72, marker='o', c=col,
                       edgecolors='white', linewidths=0.7,
                       alpha=0.95, zorder=4)
            ax.scatter(sub_cl['params'], sub_cl[metric],
                       s=140, marker='*', c='white',
                       edgecolors=col, linewidths=1.6,
                       alpha=1.0, zorder=5)

        # Reference lines for model-averaged values
        ax.axhline(v_clean, color='#2166AC', lw=1.2, ls=':', alpha=0.85,
                   zorder=1)
        ax.axhline(v_cb, color='#525252', lw=1.2, ls=':', alpha=0.85,
                   zorder=1)
        # Side labels for the references — placed inside the plot area
        ax.text(0.97, v_clean, f'clean evid. avg = {v_clean:.1f}',
                color='#2166AC', va='center', ha='right',
                fontsize=11, style='italic',
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5))
        ax.text(0.97, v_cb, f'closed-book avg = {v_cb:.1f}',
                color='#525252', va='center', ha='right',
                fontsize=11, style='italic',
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=1.5))

        ax.set_xscale('log')
        ax.set_xlabel('Model parameters (B, log scale)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.3, 1500)
        ax.set_ylim(*ylim)

    def _panel_a(self, ax):
        self._scatter_family_scaling(
            ax, 'acc', (15, 102), 'Accuracy (%)', lower_better=False)
        self.add_panel_label(ax, 'a',
                             'Accuracy scaling collapses under clean evidence')

    def _panel_b(self, ax):
        self._scatter_family_scaling(
            ax, 'high_risk', (-1, 28), 'High-risk error rate (%)',
            lower_better=True)
        self.add_panel_label(ax, 'b',
                             'High-risk error: floor effect under clean evidence')

    def _panel_c(self, ax):
        self._scatter_family_scaling(
            ax, 'dang_oc', (-1, 50), 'Dangerous overconf. rate (%)',
            lower_better=True)
        self.add_panel_label(ax, 'c',
                             'Model-side scaling does not save dang. overconf.')

    # ----- Panel d: variance decomposition (two-way ANOVA-style) -----
    def _panel_d(self, ax):
        """Decompose variance in each metric across (family, condition).
        Compute a two-way ANOVA-like sum of squares decomposition:
          SS_total = SS_family + SS_condition + SS_interaction + SS_residual.
        We use the model-level data; SS_family = sum over family of
        n_family * (family_mean - grand_mean)^2 etc."""
        metrics = [('acc', 'Accuracy'),
                   ('high_risk', 'High-risk error'),
                   ('dang_oc', 'Dangerous overconf.')]
        components = ['Condition', 'Family', 'Interaction', 'Residual']
        comp_color = {
            'Condition':   '#2166AC',
            'Family':      '#984EA3',
            'Interaction': '#F4A582',
            'Residual':    '#bdbdbd',
        }

        decomp = {}
        for key, lbl in metrics:
            data = self.df_ind[['family', 'Condition', key]].dropna().copy()
            grand = data[key].mean()
            ss_total = ((data[key] - grand) ** 2).sum()

            # Family main effect
            ss_fam = 0.0
            for fam, sub in data.groupby('family'):
                ss_fam += len(sub) * (sub[key].mean() - grand) ** 2
            # Condition main effect
            ss_cond = 0.0
            for cond, sub in data.groupby('Condition'):
                ss_cond += len(sub) * (sub[key].mean() - grand) ** 2
            # Interaction = SS(cell means) - SS_family - SS_condition
            ss_cells = 0.0
            for (fam, cond), sub in data.groupby(['family', 'Condition']):
                ss_cells += len(sub) * (sub[key].mean() - grand) ** 2
            ss_inter = max(0.0, ss_cells - ss_fam - ss_cond)
            ss_resid = max(0.0, ss_total - ss_cells)

            tot = ss_total if ss_total > 0 else 1.0
            decomp[lbl] = {
                'Family':     ss_fam / tot,
                'Condition':  ss_cond / tot,
                'Interaction': ss_inter / tot,
                'Residual':   ss_resid / tot,
            }

        labels = list(decomp.keys())
        x = np.arange(len(labels))
        width = 0.65
        bottoms = np.zeros(len(labels))
        for comp in components:
            vals = np.array([decomp[lbl][comp] for lbl in labels]) * 100
            ax.bar(x, vals, width=width, bottom=bottoms,
                   color=comp_color[comp], edgecolor='white', linewidth=0.8,
                   label=comp)
            # Inline value labels (only if segment >= 3pp)
            for xi, v, b in zip(x, vals, bottoms):
                if v >= 3:
                    ax.text(xi, b + v / 2, f'{v:.0f}',
                            ha='center', va='center', fontsize=11.5,
                            fontweight='bold',
                            color='white' if comp != 'Residual' else '#333')
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel('Variance explained (%)')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
                  ncol=4, frameon=False, fontsize=11.5,
                  handletextpad=0.5, columnspacing=1.2)
        self.add_panel_label(ax, 'd',
                             'Condition explains more variance than model family')

    # ----- Panel e: family x condition heatmap of high-risk error -----
    def _panel_e(self, ax):
        # Build matrix: rows = families (in size of family / order), cols = conditions
        mat = np.full((len(self.FAMILY_ORDER), len(self.CONDITIONS)), np.nan)
        for i, fam in enumerate(self.FAMILY_ORDER):
            for j, cond in enumerate(self.CONDITIONS):
                sub = self.df_ind[(self.df_ind['family'] == fam) &
                                  (self.df_ind['Condition'] == cond)]
                if len(sub):
                    mat[i, j] = sub['high_risk'].mean()

        im = ax.imshow(mat, aspect='auto', cmap='Reds',
                       vmin=0, vmax=20, interpolation='nearest')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = 'white' if v > 10 else '#222'
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            fontsize=12, fontweight='bold', color=txt_color)

        ax.set_xticks(range(len(self.CONDITIONS)))
        ax.set_xticklabels([self.COND_DISPLAY[c] for c in self.CONDITIONS],
                           rotation=30, ha='right', fontsize=12.5)
        ax.set_yticks(range(len(self.FAMILY_ORDER)))
        ax.set_yticklabels(self.FAMILY_ORDER, fontsize=13)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        for s in ax.spines.values():
            s.set_visible(False)

        cbar = self.fig.colorbar(im, ax=ax, fraction=0.030, pad=0.012,
                                 aspect=14)
        cbar.set_label('Family-mean high-risk error rate (%)', fontsize=12.5)
        cbar.ax.tick_params(labelsize=11)

        self.add_panel_label(ax, 'e',
                             'Column dominates row: condition outweighs family')

    # ----- Panel f: within-family std of high-risk per condition -----
    def _panel_f(self, ax):
        # x: condition; y: within-family std of high_risk
        # bars per condition; color via condition palette
        # Also overlay individual family points
        x = np.arange(len(self.CONDITIONS))
        rng = np.random.default_rng(13)

        for i, cond in enumerate(self.CONDITIONS):
            stds = []
            for fam in self.FAMILY_ORDER:
                sub = self.df_ind[(self.df_ind['family'] == fam) &
                                  (self.df_ind['Condition'] == cond)]
                if len(sub) >= 2:
                    stds.append((fam, sub['high_risk'].std()))
            if not stds:
                continue
            mean_std = np.mean([s for _, s in stds])
            ax.bar(i, mean_std, width=0.65,
                   color=self.COND_COLOR[cond], edgecolor='black',
                   linewidth=0.7, alpha=0.85, zorder=2)
            # overlay family dots
            for fam, s in stds:
                jit = rng.uniform(-0.18, 0.18)
                ax.scatter(i + jit, s, s=70, c=self.FAMILY_COLOR[fam],
                           edgecolors='white', linewidths=0.7,
                           alpha=0.92, zorder=4)
            # value text on top
            ax.text(i, mean_std + 0.25, f'{mean_std:.1f}',
                    ha='center', va='bottom', fontsize=12,
                    fontweight='bold')

        ax.set_xticks(x)
        short = {'Closed-book': 'Closed-\nbook',
                 'Clean evidence': 'Clean\nevidence',
                 'Conflict evidence': 'Conflict\nevidence',
                 'Standard RAG': 'Standard\nRAG',
                 'RaR': 'Agentic\nRAG',
                 'Max context': 'Max\ncontext'}
        ax.set_xticklabels([short[c] for c in self.CONDITIONS],
                           fontsize=12.5, ha='center')
        ax.set_ylabel('Within-family s.d. of high-risk error (pp)')
        ax.set_ylim(0, None)
        self.add_panel_label(ax, 'f',
                             'Curated evidence collapses within-family safety variance')

    # ----- Panel g: family-level Δ (closed-book → clean evidence) -----
    def _panel_g(self, ax):
        # Three metrics, grouped bars per family
        metrics = [('acc', 'Δ Accuracy', '#2166AC', False),
                   ('high_risk', 'Δ High-risk', '#D6604D', True),
                   ('dang_oc', 'Δ Dang. overconf.', '#7F0F0F', True)]
        x = np.arange(len(self.FAMILY_ORDER))
        n_m = len(metrics)
        width = 0.26

        for i_m, (key, lbl, col, lower_better) in enumerate(metrics):
            deltas = []
            for fam in self.FAMILY_ORDER:
                sub_cb = self.df_ind[(self.df_ind['family'] == fam) &
                                     (self.df_ind['Condition'] == 'Closed-book')]
                sub_cl = self.df_ind[(self.df_ind['family'] == fam) &
                                     (self.df_ind['Condition'] == 'Clean evidence')]
                if len(sub_cb) and len(sub_cl):
                    d = sub_cl[key].mean() - sub_cb[key].mean()
                    # for safety metrics, flip sign so positive = improvement
                    if lower_better:
                        d = -d
                    deltas.append(d)
                else:
                    deltas.append(np.nan)
            xs = x + (i_m - (n_m - 1) / 2) * width
            ax.bar(xs, deltas, width=width, color=col, edgecolor='white',
                   linewidth=0.6, alpha=0.92, label=lbl)

        ax.axhline(0, color='black', lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(self.FAMILY_ORDER, fontsize=12.5,
                           rotation=15, ha='right')
        ax.set_ylabel('Improvement under clean evidence (pp)')
        ax.legend(loc='upper right', frameon=False, fontsize=12,
                  handletextpad=0.5, columnspacing=1.0,
                  ncol=1)
        # explanatory note inside the panel at upper-left
        ax.text(0.02, 0.98,
                'positive = better; safety-metric signs flipped',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11, style='italic', color='#444',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=2))
        self.add_panel_label(ax, 'g',
                             'Curated evidence equalises families: weakest gain the most')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')
        print(f'Saved: {self.OUT_PNG}')


if __name__ == '__main__':
    Figure4().build().save()
