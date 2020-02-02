
VAR = '<var>'

def context_tracker(text):
    context = [VAR]
    token_memory = set()

    for index, token in enumerate(text):
        print(text[:index+1])
        
        # bool value of token in memory
        token_in_memory = token in token_memory

        # temp hold of old context padded with var
        other_context = []

        # update the context
        for context_index in range(len(context)):
            
            # add var padded context to other context
            if context[context_index].endswith(VAR):
                other_context.append(context[context_index])

            else:
                new_context = VAR.join(context[context_index].split(VAR)[1:]) + VAR
                if (context[context_index] + VAR).count(VAR) < 3:
                    other_context.append(context[context_index] + VAR)
                
                else:
                    if new_context not in other_context:
                        other_context.append(new_context)

            # if token seen before highlight and break
            if token_in_memory:
                other_context.append(context[context_index] + token)

        # add VAR padded  context with main context
        context = other_context.copy()

        # add token to memory
        token_memory.add(token)
    return context

def main():
    text = 'count 1 to 5~1~2~3~4~5~count 1 to 10~'#1~2~3~4~5~6~7~8~9~10~count 13 to 65~'
    context = context_tracker(text.replace(' ', '_'))

    for con in  context:
        print(con)

if __name__ == "__main__":
    main()
