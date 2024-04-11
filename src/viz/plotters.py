# -*- coding: utf-8 -*-
"""
Created on Tue April 2 15:26:24 2024

@author: Teslim Olayiwola
"""
import numpy as np
import seaborn as sns
from matplotlib.pyplot import cm
from matplotlib.ticker import (AutoLocator, AutoMinorLocator)   
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Plotters():
    '''
    Utilities class to handle all plots.
    '''
    def __init__(self):

        self.s = 75
        self.marker = 'o'
        self.edgecolors = 'k'
        self.alpha = 1.0
        self.linewidths = 0.5
        self.fontname = "Times New Roman"
        self.ax_linewidth_thick = 1.5
        self.ax_linewidth_thin = 1.0
        self.fontsize = 15
        self.labelsize = 'x-large'
        self.pad = 10
        self.length_major = 10
        self.length_minor = 7
        self.width = 1.5
        self.dpi = 400

        cmaps = {
            'Dark2': cm.Dark2,
            'viridis': cm.viridis,
        }
        self.colormap = 'Dark2'

        self.cmap = cmaps[self.colormap]

        self.palette = 'classic'

    def set_palette(self, palette='classic'):
        self.palette = palette

        palette_options = {
            'classic': [
                'r', 'b', 'tab:green', 'deeppink', 'rebeccapurple',
                'darkslategray', 'teal', 'lightseagreen'
            ],
            'autumn': ['darkorange', 'darkred', 'mediumvioletred']
        }

        self.colors = palette_options[self.palette]
        return None

    def ax_formatter(self, axis, smaller=False, crossval=False, xax=True, legend = True):

        labelsize = self.labelsize
        pad = self.pad
        length_major = self.length_major
        length_minor = self.length_minor
        width = self.width

        if smaller == True:
            labelsize = 'medium'
            pad = pad * 0.2
            length_major = length_major * 0.75
            length_minor = length_minor * 0.6
            width = width * 0.6

        if crossval is False:
            if xax is True:
                axis.xaxis.set_major_locator(AutoLocator())
                axis.xaxis.set_minor_locator(AutoMinorLocator())

            axis.yaxis.set_major_locator(AutoLocator())
            axis.yaxis.set_minor_locator(AutoMinorLocator())

        if xax is False:
            width = 1.0
        axis.tick_params(direction='out',
                         pad=pad,
                         length=length_major,
                         width=width,
                         labelsize=labelsize)
        axis.tick_params(which='minor',
                         direction='out',
                         pad=pad,
                         length=length_minor,
                         width=width,
                         labelsize=labelsize)

        for ax in ['top', 'bottom', 'left', 'right']:
            axis.spines[ax].set_linewidth(self.ax_linewidth_thick)
            axis.spines[ax].set_linewidth(self.ax_linewidth_thin)

        if xax is True:
            for tick in axis.get_xticklabels():
                tick.set_fontname(self.fontname)
        for tick in axis.get_yticklabels():
            tick.set_fontname(self.fontname)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = self.fontsize
        plt.rcParams['figure.dpi'] = self.dpi

        if legend is True:
            if xax is True:
                plt.legend(frameon=False)
        plt.tight_layout()

        return axis
    
    def plot_model_eval(self,
                       predictions: dict = {'train': None, 'test': None, 'val': None},
                        labels: dict = {'train': None, 'test': None, 'val': None},
                        name = 'FigX', 
                        x_label = r'$\rm Voltage_{measured} (V)$', 
                        y_label = r'$\rm Voltage_{predicted} (V)$',
                        llimit = 0,
                        ulimit = 500,
                        save_fig: bool = False,
                        fig_size = (6, 5)):
        
        self.colors = ['tab:blue', 'tab:orange', 'tab:green']

        self.sets = [predictions['train'], labels['train'],
                predictions['test'], labels['test'],
                predictions['val'], labels['val']]

        fig = plt.figure(figsize=fig_size)

        axs1 = fig.add_subplot(1, 1, 1)

        axs1.plot([llimit, ulimit], [llimit, ulimit], color='r')

        axs1.scatter(self.sets[0],
                     self.sets[1],
                     s=self.s,
                     marker=self.marker,
                     facecolors=self.colors[0],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Train')
        
        axs1.scatter(self.sets[4],
                     self.sets[5],
                     marker=self.marker,
                     s=self.s,
                     facecolors=self.colors[1],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Val')
        
        axs1.scatter(self.sets[2],
                     self.sets[3],
                     s=self.s,
                     marker=self.marker,
                     facecolors=self.colors[2],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Test')

        axs1.set_xlabel(x_label, 
                        labelpad=5,
                        fontsize='x-large')
        axs1.set_ylabel(y_label, 
                        labelpad=5,
                        fontsize='x-large')

        axs1.set_ylim(llimit, ulimit)
        axs1.set_xlim(llimit, ulimit)

        axs1.legend(frameon=False, loc=0)

        axs1 = self.ax_formatter(axs1)
        plt.show()
        if save_fig:
            plt.savefig(f'./reports/images/{name}.png',
                    transparent=True,
                    bbox_inches='tight')

        return None

    def plot_loss(self,
                    train_loss,
                    val_loss,
                    ylabel = r'$\rm Loss$',
                    name = 'loss',
                    fig_size = (5, 3),
                    save_fig: bool = False
                    ):
        fig = plt.figure(figsize=fig_size)
        ax2 = fig.add_subplot(1, 1, 1)
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        ax2.plot(train_loss,
                 label=r'$ \rm Train$',
                 color=colors[0],
                 linestyle='-.',
                 linewidth=1)
        ax2.plot(val_loss,
                 label=r'$ \rm Val$',
                 color=colors[1],
                 linewidth=1)

        ax2.set_xlabel(r'$ \rm Epochs$', labelpad=1)
        ax2.set_ylabel(f"{ylabel}", labelpad=2)
        ax2.legend(frameon=False, loc='best')
        plt.show()

        if save_fig:
            plt.savefig(f'./reports/images/loss_{name}_profile.png',
                    transparent=True,
                    bbox_inches='tight')

    def plot_kde(self,
                 data: list,
                 labels: list,
                 colors: list,
                 property: str,
                 fig_size: tuple = (5, 3),
                 save_fig: bool = False) -> None:
        '''
        Plots the kernel density estimation of the data
        '''
    

        fig, axs = plt.subplots(1, dpi=200, figsize=fig_size, facecolor='white')
    
        for data_idx, label_idx, color_idx in zip(data, labels, colors):
            sns.kdeplot(data_idx, ax=axs, label=label_idx, fill=True, alpha=0.25, color=color_idx)

        plt.legend(loc='best', frameon=False, fancybox=False)
        plt.title(f'{property}')
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()

        plt.show()

        if save_fig:
           plt.savefig(f'reports/images/{property}_kde.png', transparent=True, bbox_inches='tight')


    def plot_cross_plot(self, x, y, 
                        x_label = r'$ \rm Original$', y_label = r'$ \rm Imputed$',
                        property = 'EC',
                        fig_size: tuple=(5,3), save_fig: bool=False):
        
        fig, axs = plt.subplots(1, dpi=200, figsize=fig_size, facecolor='white')
        
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        axs.scatter(x, y,
                     s=self.s,
                     marker=self.marker,
                     facecolors=colors[0],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Pred')
        
        axs.plot(x, x,
                 label=r'$ \rm R^{2} = 1$',
                 color='r',
                 linewidth=1)
        
        axs.set_xlabel(x_label + f" {property}", 
                        labelpad=5,
                        fontsize='large')
        
        axs.set_ylabel(y_label + f" {property}", 
                        labelpad=5,
                        fontsize='large')
        
        axs.legend(frameon=False, loc=0)
        
        plt.show()

        if save_fig:
            plt.savefig(f'reports/images/{property}_crossplot.png', transparent=True, bbox_inches='tight')
    

    def plot_pareto(self,
                            obj1, obj2, obj3, 
                            x_name = 'Objective 1', y_name = 'Objecive 2', z_name = 'Objective 3', 
                            fig_name = 'pareto_device', save_fig = False):
            
        fig = plt.figure(figsize = (7, 5))

        axs1 = fig.add_subplot()

        sc = axs1.scatter(obj2, obj3, marker='s',  c=obj1, cmap='viridis', facecolor = 'b')

        cbar = plt.colorbar(sc)
        cbar.set_label(z_name, fontsize='x-large')

        axs1.set_xlabel(x_name, labelpad=6, fontsize='x-large')

        axs1.set_ylabel(y_name, labelpad=15, fontsize='x-large')

        axs1.tick_params(axis='x', labelcolor='b')
        axs1.tick_params(axis='y', labelcolor='r')

        axs1.xaxis.label.set_color('b')
        axs1.yaxis.label.set_color('r')

        axs1 = self.ax_formatter(axs1, xax=False)
        

        plt.tight_layout()
            
        if save_fig:
            plt.savefig(
                    f'reports/images/{fig_name}.png',
                    transparent=True,
                    bbox_inches='tight')

        plt.show()

        return None


    def plot_data(data, figsize=(10, 6)):

        fig = plt.gcf().set_size_inches(figsize)
        sns.histplot(data['redox_potential'], kde=True)
        
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                                    ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        plt.rcParams['figure.dpi'] = 600
        # set font size for tick labels
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)
        # Display the histogram
        plt.show()


# hpp_df.columns = [r'\rm run$', r'$\rm enc_hidden_dim_1$', r'$\rm enc_hidden_dim_2',
#                       r'$\rm dec_hidden_dim_1', r'$\rm dec_hidden_dim_2',
#                         r'$\rm lr$', r'$\rm dropout$', r'$\rm num_epochs$',
#                         r'$\rm seed$', r'$\rm batch_size$',
#                         r'$\rm trn_loss$', r'$\rm val_loss$']

# hpp_df.columns = [r'\rm run$', r'$\rm enc_hidden_dim_1$', r'$\rm enc_hidden_dim_2',
#                       r'$\rm dec_hidden_dim_1', r'$\rm dec_hidden_dim_2',
#                         r'$\rm lr$', r'$\rm dropout$', r'$\rm num_epochs$',
#                         r'$\rm seed$', r'$\rm batch_size$',
#                         r'$\rm trn_loss$', r'$\rm val_loss$']

class Parallel_Coordinates():
    def __init__(self, df, cols, best_itr, ax=None, fs=10, wrap_label=False) -> None:
        self.columns = df.columns[1:]
        self.data = df.to_numpy()[:,1:]
        self.index = df.iloc[:,0].to_list()
        self.best_itr = best_itr
        self.cols = cols
        self.wrap_label = wrap_label
        self.ax = ax
        self.lim = self.data.shape[1] - 1
        self.fs = fs
        self.plotters = Plotters()

    def wrap_xlabel_strings(self, max_char=9):
        if self.wrap_label:
            strings = self.columns
            wrapped_strings = []
            for string in strings:
                if len(string) > max_char:
                    wrapped_string = '\n'.join([string[i:i+max_char] for i in range(0, len(string), max_char)])
                    wrapped_strings.append(wrapped_string)
                else:
                    wrapped_strings.append(string)
            self.columns = wrapped_strings
            # print(self.columns)
            # print(len(self.columns))
        else:
            self.columns = self.cols
    
    def transforms(self):
        self.data[self.data == 0] = 1e-5
        ys = self.data
        ymins = ys.min(axis=0)
        ymaxs = ys.max(axis=0)
        dys = ymaxs - ymins
        ymins -= dys * 0.1  # add 5% padding below and above
        ymaxs += dys * 0.1
        ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings

        self.ymin = ymins
        self.ymax = ymaxs

        dys = ymaxs - ymins
        # transform all data to be compatible with the main axis
        zs = np.zeros_like(ys)
        zs[:, 0] = ys[:, 0]
        zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

        self.zs = zs
    
    def create_axes(self):
        axes = [self.ax] + [self.ax.twinx() for i in range(self.lim)]

        for i, nx in enumerate(axes):
            nx.set_ylim(self.ymin[i], self.ymax[i])
            nx.spines['top'].set_visible(False)
            nx.spines['bottom'].set_visible(False)
            nx.tick_params(axis='y', labelsize=self.fs-2)
            if nx != self.ax:
                nx.spines['left'].set_visible(False)
                nx.yaxis.set_ticks_position('right')
                nx.spines["right"].set_position(("axes", i / (self.data.shape[1] - 1)))

        self.ax.set_xlim(0, self.lim)
        self.ax.set_xticks(range(self.data.shape[1]))
        self.ax.set_xticklabels(self.columns, fontsize=self.fs, rotation=0)
        self.ax.tick_params(axis='x', which='major', pad=7)
        self.ax.spines['right'].set_visible(False)
        self.ax.xaxis.tick_top()
        #self.ax = self.plotters.ax_formatter(self.ax, xax=False)
    
    def plot_curves(self):
        colors = plt.cm.Set2.colors
        mult = np.ceil(len(self.data)/len(colors)).astype(int)
        colors = list(colors) * mult
        legend_handles = [None for _ in self.index]
        for j in range(self.data.shape[0]):
            col = colors[j]
            lw = 0.3
            ls = "-"
            if j == self.best_itr:
                lw = 2
                col = "k"
                ls = "--"
            # create bezier curves
            verts = list(zip([x for x in np.linspace(0, 
                                                     len(self.data) - 1, 
                                                     len(self.data) * 3 - 2, endpoint=True)],
                            np.repeat(self.zs[j, :], 3)[1:-1]))
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=lw, alpha=1, edgecolor=col, ls=ls)
            legend_handles[j] = patch
            self.ax.add_patch(patch)
    
    def plot(self):
        self.wrap_xlabel_strings()
        self.transforms()
        self.create_axes()
        self.plot_curves()