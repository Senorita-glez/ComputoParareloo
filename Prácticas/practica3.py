import time 
from multiprocessing import Pool
import math

#Nivelacion 
def nivelacion_cargas(data, n_cores):
    longitud=len(data)
    # Con esta línea se obtienen los cores que tendrán más elementos
    p_mas = len(data)%n_cores
    # Se obtienen cuántos datos le tocan a cada core sin contar los que les toca más
    numero_datos = math.floor(longitud/n_cores)
    lim_inf=[]
    lim_sup=[]
    for n_datos, i in zip([numero_datos+1 if i<p_mas else numero_datos for i in range(n_cores)], range(n_cores)):
        # para la primera parte se ponen los cores a los que les toca 1 elemento más
        if i<p_mas:
            # se multiplica por la posición en la que se encuentra el core para sacar el límite inferior
            lim_inf.append(i*n_datos)
            # Sacamos el limite superior sumando los elementos que le tocan a ese core
            lim_sup.append(lim_inf[-1]+n_datos)
        # para los cores con menos cores
        else:
            # Se suman los que les toca más
            lim_inf.append(i*n_datos+p_mas)
            # Se suman los que le tocan
            lim_sup.append(lim_inf[-1]+n_datos)
    return lim_inf, lim_sup



#Suma de fracciones por pares
def fraccSum(list):
    nlist = []
    #print(list)
    for i in range(0, len(list)):
        if i%2==0:
            nlist.append(1/list[i])
            #print(' + 1/', list[i])
        else: 
            nlist.append(-1*1/list[i])
            #print('- 1/' ,  list[i])
    return nlist


if __name__ == '__main__':
    N_CORES=1
    number  = 1000000
    cc = 0
    i = 1

    #Generar impares
    fracc = []
    if number%2 == 0:
        while cc != number+1:
            fracc.append(i)
            i +=2
            cc +=1
        
    else:
        while cc!= number:
            fracc.append(i)
            i +=2
            cc +=1
    # Obtener los límites superiores e inferiores de la lista
    lim_inf, lim_sup = nivelacion_cargas(fracc,N_CORES)

    inicio = time.time()
    with Pool(N_CORES) as p:
        vals = p.map(fraccSum, [fracc[lim_inf[i]:lim_sup[i]] for i in range(N_CORES)])
        aprox = sum(vals[0])
        print(aprox)
    print('Tiempo total de procesamiento:', time.time()-inicio)

