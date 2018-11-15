

def log(output='', title=None):
    if type(output) in [str, int, type(None)]:
        print('{} = {}'.format(title, output))
        return

    if title != None:
        print('\n{} \n{}'.format(title, ''.join(['=' for _ in range(len(title)+2)])))

    print('{}\n'.format(output))
    return

def formatFloat(number):
	return round(number, 4)