import itertools
import pandas as pd
import multiprocess
import time

df = pd.read_csv('Prácticas\Practica2\credit_dataCLEAN.csv')
df = df.head(100000)

param_grid_knn = {
    'n_neighbors': [x for x in range(1, 31, 2)],  # 1 a 30 vecinos con paso de 2
    'weights': ['uniform', 'distance'],           # Peso
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Tipos de distancias
}

keys_knn, values_knn = zip(*param_grid_knn.items())
combinations_knn = [dict(zip(keys_knn, v)) for v in itertools.product(*values_knn)]

def nivelacion_cargas(D, n_p):
    """
    """
    t = len(D) // n_p  
    r = len(D) % n_p   

    out = []
    start = 0
    for i in range(n_p):
        end = start + t + (1 if i < r else 0)
        out.append(D[start:end])
        start = end
    
    return out


def evaluate_set(df, namecolum, hyperparameter_set, lock):
    """
    Evalúa un conjunto de hiperparámetros en el KNeighborsClassifier.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = df.drop(columns=[namecolum])  
    y = df[namecolum]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    for s in hyperparameter_set:
        clf = KNeighborsClassifier()
        clf.set_params(
            n_neighbors=s['n_neighbors'], 
            weights=s['weights'], 
            metric=s['metric']
        )
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)  
        lock.acquire()
        try:
            print(f"{s} \t Accuracy:{accuracy_score(y_test, y_pred)}")
        finally:
            lock.release()



if __name__ == '__main__':
    N_THREADS = 8  # No hilos
    splits = nivelacion_cargas(combinations_knn, N_THREADS)
    for i, split in enumerate(splits):
        print(f"Proceso {i+1} tiene {len(split)} combinaciones.")
    lock = multiprocess.Lock()    
    threads = []
    for i in range(N_THREADS):
        threads.append(multiprocess.Process(target=evaluate_set, args=(df, 'Profile Score', splits[i], lock)))
    start_time = time.perf_counter()    
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")