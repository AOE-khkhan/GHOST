from copy import deepcopy
import networkx as nx

VAR = '<var>'
MEMORY = set()
TRACKS = {VAR:{}}

def context_tracker(text):
    graph = {VAR:{}}
    leaves = [(graph, VAR)]

    tracks = [[TRACKS[VAR], '']]

    for token in text:        
        # bool value of token in memory
        token_in_memory = token in MEMORY

        # temp hold of old context padded with var
        new_leaves, new_tracks = [], []

        # update the context
        for parent_node, key in leaves:
            node = parent_node[key]
            added_var = False

            # add var padded context to other context
            if key != VAR:
                node[VAR] = {}
                added_var = True
                new_node, new_node_token = (node, VAR)

            else:
                new_node, new_node_token = (parent_node, VAR)

            new_leaves.append((new_node, new_node_token))
            
            # if token seen before highlight and break
            if token_in_memory:
                node[token] = {}
                new_leaves.append((node, token))

            for track in tracks:
                last_node_checked, track_model = track

                if not token_in_memory or token not in last_node_checked:
                    last_node_checked[token] = {}
                    track_model_pad = ''

                else:
                    track_model_pad = token

                new_tracks.append([last_node_checked[token], track_model+track_model_pad])

                if not added_var:
                    continue

                if VAR not in last_node_checked:
                    last_node_checked[VAR] = {}
                    track_model_pad = ''

                else:
                    track_model_pad = VAR

                new_tracks.append([last_node_checked[VAR], track_model+track_model_pad])

        # add VAR padded  context with main context
        leaves = new_leaves.copy()
        tracks = new_tracks.copy()

        # add token to memory
        MEMORY.add(token)

    tracks = set(x[1] for x in tracks)
    return graph, tracks

def rprint(graph, indent=''):
    for key in graph:
        print(indent + key)
        rprint(graph[key], indent + '...')

def main():
    document = 'count 1 to 5~1~2~3~4~5~count 1 to 10~'#1~2~3~4~5~6~7~8~9~10~count 13 to 65~'
    sentences = document.split('~')

    for sentence in sentences:
        text = sentence.replace(' ', '_')
        graph, tracks = context_tracker(text)

        print(text)
        # rprint(graph)
        for track in tracks:
            print('  ', track)

if __name__ == "__main__":
    main()
