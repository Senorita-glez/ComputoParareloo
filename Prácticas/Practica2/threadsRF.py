import itertools
import pandas as pd
import multiprocess
import time

# Cargar el dataset
df = pd.read_csv('credit_dataCLEAN.csv')
df = df.head(100000)

# Definir la cuadrícula de hiperparámetros para Random Forest
param_grid_rf = {
    'n_estimators': [x for x in range(10, 101, 5)],  
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2'],  
    'warm_start': [True, False]
}

# Generar todas las combinaciones de hiperparámetros
keys_rf, values_rf = zip(*param_grid_rf.items())
combinations_rf = [dict(zip(keys_rf, v)) for v in itertools.product(*values_rf)]

# Función para nivelar cargas
def nivelacion_cargas(D, n_p):
    t = len(D) // n_p  
    r = len(D) % n_p   

    out = []
    start = 0
    for i in range(n_p):
        end = start + t + (1 if i < r else 0)
        out.append(D[start:end])
        start = end
    
    return out

# Función para evaluar el conjunto de hiperparámetros
def evaluate_set(df, namecolum, hyperparameter_set, lock):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = df.drop(columns=[namecolum])  
    y = df[namecolum]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
    
    for s in hyperparameter_set:
        clf = RandomForestClassifier()
        clf.set_params(
            n_estimators=s['n_estimators'], 
            criterion=s['criterion'], 
            max_features=s['max_features'], 
            warm_start=s['warm_start']
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        lock.acquire()
        try:
            print(f"{s} \t Accuracy:{accuracy_score(y_test, y_pred)}")
        finally:
            lock.release()

# Ejecución principal
if __name__ == '__main__':
    N_THREADS = 2
    
    splits = nivelacion_cargas(combinations_rf, N_THREADS)

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
