
def variable(archivo,variable_name,variable):
    with open(archivo,'a+') as file:
        for i in range(len(variable_name)):
            output=f"\n{variable_name[i]}={variable[i]}"
            print(output)
            file.write(output)
            

def testeo():
    print('funciona')
    