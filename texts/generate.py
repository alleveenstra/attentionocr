import wikipedia

for index in range(1000):
    try:
        p = wikipedia.page(wikipedia.random(1))
        p = '\n'.join([line for line in p.content.split('\n') if len(line.strip()) > 0 and not line.startswith('=')])
        with open('%d.txt' % index, 'w') as f:
            f.write(p)
    except:
        pass