def checkTests(filesCommitted): 
    stringvar_list= filesCommitted.split(',')
    string_count = len(stringvar_list)
    return string_count

st = "[u'modified', u'modified]'"

print(checkTests(st))

