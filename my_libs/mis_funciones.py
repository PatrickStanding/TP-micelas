def cifra_siginicativa(valor,error):
    error=abs(error)
    error_str=f"{error:.1g}" #1 ciffra cignificativa
    if error<1:
        cifra_sign=error_str.split('.')[1]
        cifra_sign=len(cifra_sign)
    elif error<10:
       cifra_sign=-1
       numero=int(round(valor,cifra_sign))
       return f"{numero} +/- {error_str}"    

    else:
        cifra_sign=cifra_sign.split('e+')
        cifra_sign=cifra_sign.replace('0','-')
    numero=round(valor,cifra_sign)
    return f"{numero} +/- {error_str}"    
    