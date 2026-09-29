import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt


# Graph, Use Case: Emergency Supply Robot

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}


hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },

    "Main_Corridor": {
        "Nursing_Station": 2.2
    },

    "Patient_Wing": {
        "Laboratory": 5.0
    },

    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },

    "Laboratory": {
        "Emergency_Ward": 3.2
    },

    "Emergency_Ward": {}
}


# Heuristic

def heuristic(current, goal):

    x1 = locations[current][0]
    y1 = locations[current][1]

    x2 = locations[goal][0]
    y2 = locations[goal][1]

    distance = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    return round(distance, 1)


# Path reconstruction

def reconstruct_path(came_from, current):

    path = []

    while current is not None:

        path.append(current)

        current = came_from[current]

    path.reverse()

    return path


# GBFS

def gbfs(start, goal):

    queue = []

    visited = set()

    expansion_order = []

    parent = {
        start: None
    }

    # Start with the heuristic value
    heapq.heappush(
        queue,
        (heuristic(start, goal), start)
    )

    while queue:

        h, current = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)

        expansion_order.append(current)

        # Goal reached
        if current == goal:

            path = reconstruct_path(
                parent,
                current
            )

            # Calculate total path cost
            cost = 0

            for i in range(len(path) - 1):

                current_node = path[i]
                next_node = path[i + 1]

                cost += hospital_graph[current_node][next_node]

            return path, cost, expansion_order

        # Explore neighbors
        for neighbor in hospital_graph[current]:

            if neighbor not in visited:

                parent[neighbor] = current

                heapq.heappush(
                    queue,
                    (
                        heuristic(neighbor, goal),
                        neighbor
                    )
                )

    return None, None, expansion_order


# A*

def a_star(start, goal):

    queue = []

    expansion_order = []

    # g(n) = cost from start
    g_cost = {
        start: 0
    }

    # Store previous node
    parent = {
        start: None
    }

    # f(n) = g(n) + h(n)
    f = (
        g_cost[start]
        + heuristic(start, goal)
    )

    heapq.heappush(
        queue,
        (f, start)
    )

    while queue:

        f, current = heapq.heappop(queue)

        expansion_order.append(current)

        # Goal reached
        if current == goal:

            path = reconstruct_path(
                parent,
                current
            )

            cost = g_cost[goal]

            return path, cost, expansion_order

        # Explore neighbors
        for neighbor, weight in hospital_graph[current].items():

            new_g = g_cost[current] + weight

            # Better path found
            if neighbor not in g_cost or new_g < g_cost[neighbor]:

                g_cost[neighbor] = new_g

                parent[neighbor] = current

                # f(n) = g(n) + h(n)
                f = (
                    g_cost[neighbor]
                    + heuristic(neighbor, goal)
                )

                heapq.heappush(
                    queue,
                    (f, neighbor)
                )

    return None, None, expansion_order


##########################################
# Streamlit GUI Code
##########################################


# Set Page Config

st.set_page_config(
    page_title="Emergency Supply Robot",
    page_icon="🏥",
    layout="wide"
)


# Title and description

st.title("Emergency Supply Robot")

st.write(
    "Use Greedy Best-First Search or A* Search "
    "to find a path through the hospital."
)


# Define the nodes and their coordinates

nodes = list(hospital_graph.keys())


# Create selectbox for start node

start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)


# Create selectbox for goal node

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)


# Create selectbox for search algorithm

algorithm = st.selectbox(
    "Select Search Algorithm",
    ["GBFS", "A*"]
)


# Run search

if st.button("Run Search"):

    if algorithm == "GBFS":

        # Run GBFS
        path, cost, expansion_order = gbfs(
            start,
            goal
        )

    else:

        # Run A*
        path, cost, expansion_order = a_star(
            start,
            goal
        )


    # Check if path was found

    if path is None:

        st.error(
            "No path was found between the selected nodes."
        )

    else:

        # Display result

        st.subheader("Search Result")

        st.write(
            f"Algorithm: {algorithm}"
        )

        st.write(
            f"Solution Path: {' → '.join(path)}"
        )

        st.write(
            f"Total Path Cost: {cost:.2f}"
        )

        st.write(
            f"Expansion Order: "
            f"{' → '.join(expansion_order)}"
        )


        # Visualize NetworkX graph

        G = nx.DiGraph()


        # Add weighted edges

        for node, neighbors in hospital_graph.items():

            for neighbor, weight in neighbors.items():

                G.add_edge(
                    node,
                    neighbor,
                    weight=weight
                )


        pos = locations


        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        # Draw nodes

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="lightblue",
            node_size=2500,
            ax=ax
        )


        # Draw all edges

        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            ax=ax
        )


        # Draw node labels

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax
        )


        # Draw edge weights

        edge_labels = nx.get_edge_attributes(
            G,
            "weight"
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax
        )


        # Create solution path edges

        path_edges = []

        for i in range(len(path) - 1):

            path_edges.append(
                (path[i], path[i + 1])
            )


        # Highlight solution path

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            edge_color="red",
            width=3,
            arrows=True,
            ax=ax
        )


        ax.set_title(
            f"{algorithm} Solution Path"
        )

        ax.axis("off")

        st.pyplot(fig)