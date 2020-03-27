import re


with open("test_manual_generation.html",'w') as out_file:
    out_file.write('<pre>')
    game_nr = 1
    for input, output in zip(open("data/test_manual_long.input"), open("test_manual_generation.txt")):
        line = 'E: '+input
        if '<type>result' in input:
            line = ('--- %d.' % game_nr)+(' %s - %s ---\n\n' % re.findall("<home> (.+) </home> <guest> (.+) </guest>",line)[0]) +line
            game_nr += 1
        line = re.sub("<length>\w+</length>", "", line)
        line = line.replace('>','&gt;').replace('<','&lt;')
        line = re.sub(r" ?&lt;(\w+)&gt; ?", r"</b> \1:<b>", line)
        line = re.sub(r"(&lt;/\w+&gt;)",r"", line)
        line = "<b>"+line+"</b>"
        out_file.write(line)#+'\n')
        out_file.write(output+'\n')
    out_file.write('</pre>')
