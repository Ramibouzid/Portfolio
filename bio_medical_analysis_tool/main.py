import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pyvis.network import Network
import matplotlib.pyplot as plt
import PyBoolNet
import openpyxl

#PyBoolNet.Tests.Dependencies.run()


# Reading excel files and convert them to dataframes
data = pd.read_excel(r'C:\Users\Orelson\PycharmProjects\Test_gamma\data.xls').iloc[:, : 3]
gene_expression = pd.read_excel(r'C:\Users\Orelson\PycharmProjects\Test_gamma\gene_experssion.xlsx').iloc[:, : 2]
output_workbook = r'C:\Users\Orelson\Desktop\output.xlsx'
f = open("Bol_equations.txt", "w+")


# Function to find gene expression of each node generate expression data column and create new cleaned dataframe
def new_expression_data(dataa, a_source, a_expression):
    dataa.rename(columns={a_source: 'Node'}, inplace=True)
    df_3 = pd.merge(dataa, gene_expression[['Node', 'logFC']], on='Node', how='left')
    df_3.rename(columns={'Node': a_source}, inplace=True)
    df_3 = df_3.replace(np.nan, 0, regex=True)
    df_3.rename(columns={'logFC': a_expression}, inplace=True)
    df_3[a_expression] = round(np.sign(df_3[a_expression]), 1)
    return df_3


# df_4 : my final dataframe
c = new_expression_data(data, 'Source', "Source_expression")
df_4 = new_expression_data(c, 'Target', "Target_expression")

# create Validation column, add it to df_4 and then output in excel file
multiply_col = df_4["Source_expression"]*df_4["Target_expression"]
df_4['multiply'] = multiply_col
df_4['validity'] = np.where(df_4['multiply'] == df_4['Interaction Type'], 'yes', 'no')
del df_4['multiply']
df_4.to_excel(output_workbook, index=False)

# print(df_4)


################ generate Boolean Equations ###################

df_5 = df_4[df_4["validity"] == "yes"]
df_5 = df_5.reset_index(drop=True)
print(df_5)
operator1 = ""
operator2 = ""
listo = []
for i in range(len(df_5)):
    if df_5["Interaction Type"][i] == -1:
        operator1="not"
    else:
        operator1 = ""
    ch = f"{df_5['Target'][i]}*={operator1} {df_5['Source'][i]}"
    for j in range(0, len(df_5)):
        if j != i :
            if df_5['Target'][j]==df_5['Target'][i]:
                if df_5["Interaction Type"][j] == -1:
                    operator2 = "not"
                else:
                    operator2 = ""
                if df_5["Source_expression"][i]*df_5["Source_expression"][j]==df_5['Target_expression'][i]:
                    ch += f" and {operator2} {df_5['Source'][j]}"
                else:
                    ch += f" or {operator2} {df_5['Source'][j]}"
    ch += "\n"
    if df_5['Target'][i] not in listo:
        f.write(ch)
    listo.append(df_5['Target'][i])

f.close()


# Build the Network
G = nx.from_pandas_edgelist(df_5, 'Source', 'Target')
pos = nx.random_layout(G)
nodes= G.nodes.data()
edges = G.edges.data()
print("nodePos : \n" , pos)
nx.draw_networkx(G, with_labels = True,pos=pos)
#nx.draw(G, with_labels = True,pos=pos, edge_color="red")

# Plot the network
nt = Network("800px", "1500px", directed=False)
nt.from_nx(G)
nt.show_buttons()
#filter_=['physics']
#nt.barnes_hut()
nt.show("nx1.html")


################## steady state test ##################
# def read_rules_text(model_name):
#     '''
#     Reads in the Boolean model
#     '''
#     rules_file = model_name
#     with open(rules_file, 'r') as fi:
#         rules = fi.read()
#     return rules
#
#
# def get_stable_motif_successions_recursive(rules, node_substitutions, succession):
#     '''
#     Recursively builds lines of successively stabilising trap spaces based on the
#     get_maximal_trap_spaces_after_percolation() function
#     '''
#
#     new_rules, maxts = get_maximal_trap_spaces_after_percolation(rules, node_substitutions)
#     if len(maxts) == 0:
#         yield succession
#     else:
#         for SM in maxts:
#             yield from get_stable_motif_successions_recursive(new_rules, SM, succession + [SM])
#
#
# def get_maximal_trap_spaces_after_percolation(rules, node_substitutions):
#     '''
#     Given a set of Boolean rules, substitutes the values given in node substitutions and after finding the
#     prime implicants it further simplifies the model by finding the LDOI of the substitution using
#     PyBoolNet.PrimeImplicants.percolate_and_remove_constants(primes) and further reducing the rules.
#     Finally it finds the stable motifs (maximal trap spaces) using
#     PyBoolNet.AspSolver.trap_spaces(primes, "max")
#
#     '''
#     rules_new = rules[:]
#     if len(node_substitutions) == 0:
#         rules_new = rules[:]
#     else:
#         for node, value in node_substitutions.items():
#             rules_new = '\n'.join([s for s in rules_new.split('\n') if s.split(',')[0] != node])
#             rules_new = rules_new.replace(node, str(value))
#     primes = PyBoolNet.FileExchange.bnet2primes(rules_new)
#     if len(rules_new.strip()) == 0:
#         return '', []
#
#     constants = PyBoolNet.PrimeImplicants.percolate_and_remove_constants(primes)
#     for line in rules_new.strip().split('\n'):
#         node = line.split(',')[0].strip()
#         rule = line.split(',')[1].strip()
#         if node in constants.keys():
#             rules_new = rules_new.replace(line + '\n', '')
#         for n, v in constants.items():
#             rule_new = rule.replace(n, str(v))
#             rules_new = rules_new.replace(rule, rule_new)
#     primes = PyBoolNet.FileExchange.bnet2primes(rules_new)
#     maxts = PyBoolNet.AspSolver.trap_spaces(primes, "max")
#
#     return rules_new, maxts
#
#
# def get_final_state_corresponding_to_trap_space_succession(rules, succession):
#     '''
#     Based on a succession line it finds the final steady state and/or oscillating components.
#     We use this function in order to not burden the recursion with storing what has been substituted already
#     and we infer the final steady state once the succession lines were built.
#
#     If a node oscillates by the end it will assign the string "osc" to the corresponding variable in
#     the returned dictionary
#     '''
#
#     node_substitutions = {}
#     for mts in succession:
#         for n, v in mts.items():
#             node_substitutions[n] = str(v)
#     new_lines = []
#     for line in rules.strip().split('\n'):
#         n, rule = line.split(',')
#         for node, value in node_substitutions.items():
#             rule = rule.replace(node, str(value))
#         new_lines.append(n + ',\t' + rule)
#     rules_new = '\n'.join(new_lines)
#     primes = PyBoolNet.FileExchange.bnet2primes(rules_new)
#     steady = PyBoolNet.AspSolver.steady_states(primes)
#     constants = PyBoolNet.PrimeImplicants.percolate_and_remove_constants(primes)
#     if len(steady) == 1:
#         return steady[0]
#     elif len(steady) == 0:  # we are dealing with an oscillation
#         nodes = [s.split(',')[0].strip() for s in rules.split('\n') if s.strip() != '']
#         final_state = {}
#         for n, v in node_substitutions.items():
#             final_state[n] = str(v)
#             if n in nodes:
#                 nodes.remove(n)
#         for n, v in constants.items():
#             final_state[n] = str(v)
#             if n in nodes:
#                 nodes.remove(n)
#         for n in nodes:
#             final_state[n] = 'osc'
#         return final_state
#     else:
#         raise ValueError('The succession does not lead to a unique attractor.')
#
#
#
# def edge_list_from_succession_line(succession):
#     '''Transfroms the succession line into an edge list'''
#
#     x = []
#     for k in succession:
#         s = ''
#         for node, state in sorted(k.items()):
#             s += '%s=%s;' % (node, str(state))
#         x.append(s)
#     return zip(x[:-1], x[1:])
#
# model_='Bol_equations.txt'
#
# model_='Bol_equations.txt'
#
# rules=read_rules_text(model_)
#
# nodes= [s.split('*=')[0].strip() for s in rules.split('\n') if s.strip()!='']
#
# print(nodes)
#
# rules= rules.replace(' *=',',\t').replace('*=',',\t').replace('not ','!').replace(' and ',' & ').replace(' or ',' | ')
#
# print(rules)
#
# starting_substitution={}
# successions=list(get_stable_motif_successions_recursive(rules,starting_substitution,[]))
# G_succession=nx.DiGraph()
# for succession in successions:
#     #fss is the final steady state (or oscillation that will be a leaf in our graph)
#     fss=get_final_state_corresponding_to_trap_space_succession(rules,succession)
#     print (succession+[fss])
#     el=list(edge_list_from_succession_line(succession+[fss]))
#     G_succession.add_edges_from(el)

# plt.figure(figsize=(12,10))
# from networkx.drawing.nx_agraph import graphviz_layout
# pos =graphviz_layout(G_succession, prog='dot')
# nx.draw(G_succession, pos, with_labels=True, arrows=True)

