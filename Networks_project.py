import networkx as nx
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from operator import itemgetter
import collections
from numpy import random, sqrt
from networkx.algorithms import community
from networkx.algorithms import distance_measures
import itertools

file1_path = 'C:/Users/szmat/Desktop/Datasets/soc-anybeat/soc-anybeat.edges'
file2_path = 'C:/Users/szmat/Desktop/Datasets/soc-hamsterster/test.txt'
file3_path = 'C:/Users/szmat/Desktop/Datasets/soc-dolphins/dolphins.txt'

g1 = nx.read_edgelist(file1_path, create_using=nx.Graph(), nodetype=int)
g2 = nx.read_edgelist(file2_path, create_using=nx.Graph(), nodetype=int)
g3 = nx.read_edgelist(file3_path, create_using=nx.Graph(), nodetype=int)
g4 = nx.karate_club_graph()


def print_basic_metrics(G):
    print('----------------------')
    print(nx.info(G))
    print('Density: ', nx.density(G))
    connected = nx.is_connected(G)
    print('Is connected: ', connected)
    print('Radius: ', nx.radius(G))

    print('Global clustering coefficient: ', nx.transitivity(G))
    print('Average clustering coefficient: ', nx.average_clustering(G))
    print('Clustering: ', nx.clustering(G))

    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
    print("Top 5 nodes by degree:")
    for d in sorted_degree[:5]:
        print(d)

    if not connected:
        components = nx.connected_components(G)
        largest_component = max(components, key=len)
        G_sub = G.subgraph(largest_component)
        print('----------------------')
        print('The largest subgraph')
        print(nx.info(G_sub))

    print('----------------------')


def plot_distribution_degrees(G, plot=False):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.axes([0.4, 0.4, 0.5, 0.5])
    plt.axis("off")
    plt.show()


def plot_community_detection(G):
    partition = community_louvain.best_partition(G)
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def infected_nodes(G, starting_points, depth):
    if depth == 0: return starting_points, len(starting_points), len(starting_points) / G.number_of_nodes()
    X = set(starting_points)
    i = 0
    while i <= depth:
        for j in starting_points:
            X = set.union(X, nx.descendants_at_distance(G, j, i))
        i += 1

    return X, len(X), len(X) / G.number_of_nodes()


def sorted_degrees(G):
    degree_dict = dict(G.degree(G.nodes()))
    return sorted(degree_dict.items(), key=itemgetter(1), reverse=True)


def information_spread(G, starting_points):
    len_x = 0
    i = 0
    while len_x < G.number_of_nodes():
        x, len_x, x_prop = infected_nodes(G, starting_points, i)
        i += 1
        print(i - 1, len_x, x_prop, x_prop >= 0.5)


def spread_from_start(G, starting_point):
    print('----------------')
    print('Starting points: ', starting_point)
    print('----------------')
    information_spread(G, starting_point)


def first_match_comm_degree(comm, dict_degree):
    l_temp = []
    first_elements = [a_tuple[0] for a_tuple in dict_degree]
    for i in comm:
        set_y = set(i)
        l_temp.append(next((a for a in first_elements if a in set_y), None))
    return l_temp


def information_spread_value(G, starting_points):
    len_x = 0
    i = 0
    temp_sum = 0
    while len_x < G.number_of_nodes():
        x, len_x, x_prop = infected_nodes(G, starting_points, i)
        i += 1
        if i > 1:
            temp_sum += x_prop
    return 1 / (sqrt(len(starting_points)) * (i ** 2)) * temp_sum


print('############################')
print('#### Basic statistics #####')
print('############################')
print_basic_metrics(g3)
print_basic_metrics(g4)

# plot_distribution_degrees(g3)
# plot_distribution_degrees(g4)

# plot_community_detection(g3)
# plot_community_detection(g4)


print('############################')
print('########## Karate ##########')
print('############################')
G = g4.copy()

# Check results twice for one randomly picked node
print('------- Random node -------')
random.seed(122)
rand_temp = random.choice(range(G.number_of_nodes()))
spread_from_start(G, [rand_temp])
print('Spread coefficient:', information_spread_value(G, [rand_temp]))

print('------- Random node -------')
random.seed(124)
rand_temp = random.choice(range(G.number_of_nodes()))
spread_from_start(G, [rand_temp])
print('Spread coefficient:', information_spread_value(G, [rand_temp]))

print('--- Highest degree node ---')
print('Nodes degrees: ', sorted_degrees(G)[:5])
spread_from_start(G, [sorted_degrees(G)[0][0]])
print('Spread coefficient:', information_spread_value(G, [sorted_degrees(G)[0][0]]))

print('------ Central point ------')
print('-- Global highest degree --')
for i in distance_measures.center(G):
    spread_from_start(G, [i])
    print('Spread coefficient:', information_spread_value(G, [i]))

print('######### Community ########')
comp = community.girvan_newman(G)
t = tuple(sorted(c) for c in next(comp))
print('2 Communities: ', t)
print('-- Global highest degree --')
temp = first_match_comm_degree(t, sorted_degrees(G))
spread_from_start(G, temp)
print('Spread coefficient:', information_spread_value(G, temp))

print('-- Local highest degree  --')
temp = [sorted_degrees(G.subgraph(t[0]))[0][1], sorted_degrees(G.subgraph(t[1]))[0][1]]
spread_from_start(G, temp)
print('Spread coefficient:', information_spread_value(G, temp))

print('----- Dominating set  -----')
D = nx.dominating_set(G)
spread_from_start(G, D)
print('Spread coefficient:', information_spread_value(G, temp))

print('---- Optimal strategy  ----')


def optimal_strategy(G):
    D = nx.dominating_set(G)

    def best_points_in_community(comm, G):
        degrees = sorted_degrees(G)
        l_temp = []
        for i in comm:
            D = G.subgraph(i)
            central_pts = distance_measures.center(D)
            first_elements = [a_tuple[0] for a_tuple in degrees]
            set_y = set(central_pts)
            l_temp.append(next((a for a in first_elements if a in set_y), None))
        return l_temp

    value = 0
    for j in range(1, len(D)):
        if j == 1:
            t = (G.nodes(),)
        else:
            comp = community.girvan_newman(G)
            limited = itertools.takewhile(lambda c: len(c) <= j, comp)
            for communities in limited:
                t = tuple(sorted(c) for c in communities)

        temp = best_points_in_community(t, G)
        spread_from_start(G, temp)
        if value - information_spread_value(G, temp) > 0.01:
            print('Spread coefficient:', information_spread_value(G, temp))
            break
        value = information_spread_value(G, temp)
        print('Spread coefficient:', value)


optimal_strategy(G)

print('############################')
print('######### Dolphins #########')
print('############################')

G = g3.copy()

print('--- Highest degree node ---')
print('Nodes degrees: ', sorted_degrees(G)[:5])
spread_from_start(G, [sorted_degrees(G)[0][0]])
print('Spread coefficient:', information_spread_value(G, [sorted_degrees(G)[0][0]]))

print('----- Dominating set  -----')
D = nx.dominating_set(G)
spread_from_start(G, D)
print('Spread coefficient:', information_spread_value(G, temp))

print('---- Optimal strategy  ----')
optimal_strategy(G)