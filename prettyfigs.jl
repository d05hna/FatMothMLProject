using CSV, DataFrames, GLMakie, AlgebraOfGraphics
GLMakie.activate!()
theme = theme_dark()
theme.textcolor=:white
theme.ytickfontcolor=:white
theme.fontsize= 16
theme.gridcolor=:white
theme.gridalpga=1
theme.palette = (color = [:turquoise,:coral,:magenta],)
# theme.palette = (color = paired_10_colors)
set_theme!(theme)

## Getting The data to Plot and Organizing Stuff

shaps = CSV.read("shapimportance.csv",DataFrame)
Mis = CSV.read("MI_results.csv",DataFrame)
select!(Mis,Not(:wb))

# Pivot around the MI Data for the format I want 
Mis.row_num = 1:nrow(Mis)
longmi = stack(Mis,Not(:row_num))
longmi = rename(longmi, :variable => :feature, :value => :values)
longmi.values = longmi.values ./ maximum(longmi.values) #Shap values are max normalized so I do that for MI too 

mi_reshape = unstack(longmi, :feature, :row_num, :values)
rename!(mi_reshape, [i => Symbol("k_$(i)") for i in 2:8])

all_values = leftjoin(shaps,mi_reshape,on=:feature)

# Helper Function to make the figures 
function make_comp_fig(d4plt,title)
    fig = Figure(figsize=(1200,400))

    ax = Axis(fig[1,1],
        xlabel="Feature",
        ylabel="Relative Importance",
        xticks = (1:length(d4plt.feature),d4plt.feature),
        xticklabelrotation=π/4,
        title = title
    )

    barplot!(ax,repeat(1:length(d4plt.feature),3),
        vcat(d4plt.boost_importance,d4plt.rbf_importance,d4plt.k_4),
        dodge = repeat(1:3,inner=length(d4plt.feature)),
        color = repeat([:coral,:turquoise,:orchid2],inner=length(d4plt.feature))
        )

    elements = [PolyElement(color = c) for c in [:coral, :turquoise, :orchid2]]
    labels = ["Gradient Boosting", "Nonlinear SVM", "Mutual Information"]
    Legend(fig[1,2], elements, labels)

    return(fig)
end

## Joy Used K = 4 for her MI stuff and all our values are basically the same from k = 2:8 so lets use 4 only 
musnames = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
d4plt = select(all_values,[:feature,:boost_importance,:rbf_importance,:k_4])

countid = [occursin("count",s) for s in d4plt.feature]
countdf = d4plt[countid,:]
countdf.feature = musnames

phaseid = [occursin("1",s) for s in d4plt.feature]
phasedf = d4plt[phaseid,:]
phasedf.feature = musnames

countfig = make_comp_fig(countdf,"Spike Count")
phasefig = make_comp_fig(phasedf,"First Spike Timing")

save("Figs/ComparingImportanceCount.png",countfig,px_per_unit=4)
save("Figs/ComparingImportancePhase.png",phasefig,px_per_unit=4)
