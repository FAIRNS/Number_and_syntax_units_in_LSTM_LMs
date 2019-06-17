import os
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sentence = 'The \\textbf{boy(s)} near the \\underline{car(s)} \\textbf{greet(s)} the'.split()

# nouns and verb positions
N1=1
N2=4
V=5
off = .15
PP_off= off+off/2
PS_off = off/2
SP_off= -off/2
SS_off = -off-off/2
SY_off = 0
no_legend = True

max_off = max(PP_off, PS_off, SP_off, SS_off)

# define series values
SS_sugg = [0 + SS_off]*len(sentence)
SS_sugg[N1] = -1 + SS_off
SS_sugg[N2] = -1 + SS_off
PS_sugg = [0 + PS_off]*len(sentence)
PS_sugg[N1] = 0 + PS_off
PS_sugg[N2] = -1 + PS_off
SP_sugg = [0 + SP_off]*len(sentence)
SP_sugg[N1] = -1 + SP_off
SP_sugg[N2] = 0 + SP_off
PP_sugg = [0 + PP_off]*len(sentence)
PP_sugg[N1] = 0 + PP_off
PP_sugg[N2] = 0 + PP_off

# Hypothesis mask
HY_sugg = [0] * len(sentence)
HY_sugg[N1] = 1
HY_sugg[N2] = 1

SS_input = [0 + SS_off]*len(sentence)
SS_input[N1] = 1 + SS_off
SS_input[N2] = 0 + SS_off
PS_input = [0 + PS_off]*len(sentence)
PS_input[N1] = 0 + PS_off
PS_input[N2] = 0 + PS_off
SP_input = [0 + SP_off]*len(sentence)
SP_input[N1] = 1 + SP_off
SP_input[N2] = 0 + SP_off
PP_input = [0 + PP_off]*len(sentence)
PP_input[N1] = 0 + PP_off
PP_input[N2] = 0 + PP_off

# Hypothesis mask
HY_input = [1] * len(sentence)
HY_input[0] = 0
HY_input[-1] = 0

SS_forget = [1 + SS_off]*len(sentence)
SS_forget[N1] = 0 + SS_off
SS_forget[N1-1] = 0 + SS_off
SS_forget[-1] = 0 + SS_off
PS_forget = [1 + PS_off]*len(sentence)
PS_forget[N1] = 0 + PS_off
PS_forget[N1-1] = 0 + PS_off
PS_forget[-1] = 0 + PS_off
SP_forget = [1 + SP_off]*len(sentence)
SP_forget[N1] = 0 + SP_off
SP_forget[N1-1] = 0 + SP_off
SP_forget[-1] = 0 + SP_off
PP_forget = [1 + PP_off]*len(sentence)
PP_forget[N1] = 0 + PP_off
PP_forget[N1-1] = 0 + PP_off
PP_forget[-1] = 0 + PP_off

# Hypothesis mask
HY_forget = [1] * len(sentence)
HY_forget[0] = 0
HY_forget[-1] = 0


SS_cell = [-1 + SS_off]*len(sentence)
SS_cell[N1-1] = 0 + SS_off
SS_cell[-1] = 0 + SS_off
PS_cell = [0 + PS_off]*len(sentence)
PS_cell[N1] = 0 + PS_off
PS_cell[N1-1] = 0 + PS_off
PS_cell[-1] = 0 + PS_off
SP_cell = [-1 + SP_off]*len(sentence)
SP_cell[N1-1] = 0 + SP_off
SP_cell[-1] = 0 + SP_off
PP_cell = [0 + PP_off]*len(sentence)
PP_cell[N1] = 0 + PP_off
PP_cell[N1-1] = 0 + PP_off
PP_cell[-1] = 0 + PP_off

HY_cell = [1] * len(sentence)
HY_cell[0] = 0
HY_cell[-1] = 0

SY_cell = [0 + SY_off] * len(sentence)
for i in range(N1+1, V):
    SY_cell[i] = 1

SS_output = [0 + SS_off]*len(sentence)
SS_output[N2] = 1 + SS_off
SS_output[-1] = 0 + SS_off
PS_output = [0 + PS_off]*len(sentence)
PS_output[N2] = 1 + PS_off
SP_output = [0 + SP_off]*len(sentence)
SP_output[N2] = 1 + SP_off
PP_output = [0 + PP_off]*len(sentence)
PP_output[N2] = 1 + PP_off

HY_output = [0] * len(sentence)
HY_output[N2] = 1

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot(ax, series, HY_series, args):
    label = args['label']
    color = args['color']
    del args['label']
    label_set = False
    # draw bordeline conditions
    series = [series[0]] + series + [series[-1]]
    HY_series = [HY_series[0]] + HY_series + [HY_series[-1]]
    for h in range(len(HY_series)):
        if 'label' in args and label_set:
            del args['label']

        if HY_series[h] == 1:
            args['color'] = color
            if not label_set:
                label_set = True
                args['label'] = label
        else:
            args['color'] = lighten_color(color, 0.2)
        
        x = []
        y = []
        if h > 0:
            x.append(h-1-0.5)
            y.append((series[h-1] + series[h])/2.0)
        x.append(h-1)
        y.append(series[h])
        if h < len(series)-2:
            x.append(h-1+0.5)
            y.append((series[h] + series[h+1])/2.0)


        ax.plot(x, y, **args)
        #ax.plot([x1,x2], [series[x1], series[x2]], **args)

        if 'label' in args:
            del args['label']
    #ax.plot(series, **args)

def plot_all_series(ax, PP_series, SS_series, PS_series, SP_series, HY_series):
    C0='#08d9d6'
    C1='#B22044'
    plot(ax, SS_series, HY_series, {'ls': '-', 'lw': 1.5, 'label': r'\textbf{Singular}-\underline{Singular}', 'color': C0})
    plot(ax, PP_series, HY_series, {'ls': '-', 'lw': 1.5, 'label': r'\textbf{Plural}-\underline{Plural}', 'color': C1})
    plot(ax, SP_series, HY_series, {'ls': '-.', 'lw': 1.5, 'label': r'\textbf{Singular}-\underline{Plural}', 'color': C0})
    plot(ax, PS_series, HY_series, {'ls': '-.', 'lw': 1.5, 'label': r'\textbf{Plural}-\underline{Singular}', 'color': C1})
    ax.set_xlim([-.5, len(HY_series)-.5])
    ax.set_xticks(range(len(HY_series)))
    ax.grid(c='w', ls='-', lw=1)
    plt.setp(ax.get_xticklabels(), visible=False)

FC='#ffffff'
plt.figure(figsize=(10,5))
fig, axs = plt.subplots(5, 1, subplot_kw={'fc':FC}) 
(sugg_ax, input_ax, forget_ax, cell_ax, output_ax) = axs
plot_all_series(sugg_ax, PP_sugg, SS_sugg, PS_sugg, SP_sugg, HY_sugg)
sugg_ax.set_ylabel(r"$\tilde{C_t}$", fontsize=24, rotation='horizontal', ha='center', va='center')
lims = [-1.5, 1.5]
ticks = lims
sugg_ax.set_yticks(lims)
sugg_ax.set_ylim([lims[0] - max_off, lims[1] +max_off])

plot_all_series(input_ax, PP_input, SS_input, PS_input, SP_input, HY_input)
input_ax.set_ylabel("$i_t$", fontsize=24, rotation='horizontal', ha='center', va='center')
ticks = [0, 1]
#lims = [0.1-max_off, 0.9+max_off]
lims = [0-max_off/2., 1 + max_off/2.]
input_ax.set_yticks(ticks)
input_ax.set_ylim([lims[0] - max_off, lims[1] +max_off])
input_ax.set_xticks([])

plot_all_series(forget_ax, PP_forget, SS_forget, PS_forget, SP_forget, HY_forget)
forget_ax.set_ylabel("$f_t$", fontsize=24, rotation='horizontal', ha='center', va='center')
ticks = [0, 1]
lims = [0-max_off/2., 1 + max_off/2.]
forget_ax.set_yticks(ticks)
forget_ax.set_ylim([lims[0] - max_off, lims[1] +max_off])
forget_ax.set_xticks([])

plot_all_series(cell_ax, PP_cell, SS_cell, PS_cell, SP_cell, HY_cell)
#cell_ax.plot(SY_cell, ls='--', lw=2, label=r'Syntax Unit', color='green')
cell_ax.set_ylabel("$C_t$", fontsize=24, rotation='horizontal', ha='center', va='center')
lims = [-1.5, 1.5]
ticks = lims
cell_ax.set_yticks(ticks)
cell_ax.set_ylim([lims[0] - max_off, lims[1] + max_off])

plot_all_series(output_ax, PP_output, SS_output, PS_output, SP_output, HY_output)
output_ax.set_ylabel("$o_t$", fontsize=24, rotation='horizontal', ha='center', va='center')
ticks = [0, 1]
lims = [-0.05-max_off, 1.05 + max_off]
output_ax.set_yticks(ticks)
output_ax.set_ylim([lims[0] - max_off, lims[1] +max_off])
output_ax.set_xticks([])

for ax in axs:
    ax.tick_params(labelsize=10)
#forget_ax.set_xticks([])
#plt.plot(forget, ls=':', lw=4, label='The girl/girls $f_t$', color='C2')
plt.xticks(ticks=range(len(sentence)), labels=sentence, fontsize=17, rotation=0)
plt.setp(fig.gca().get_xticklabels(), visible=True)
handles, labels = cell_ax.get_legend_handles_labels()
if not no_legend:
    legend = fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9)

path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(path, 'unit-timeseries-cartoon.pdf')
print("Saving to {}".format(output_path))
fig.align_ylabels(axs)
plt.tight_layout()
fig.savefig(output_path)
