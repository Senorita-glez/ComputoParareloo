# Importing the threading module
import threading, multiprocess, time, random, multiprocessing

# Function to add profit to the deposit
def incrementar(lock, shared): 
    with lock:
        shared.value = shared.value + 1
        print(shared.value)

if __name__=='__main__':
    # Now we will evaluated with more threads
    threads=[]
    N_THREADS=12
    shared = multiprocessing.Value('i', 0)
    lock=multiprocessing.Lock()

    for i in range(N_THREADS):
        # Se generan los hilos de procesamiento
        threads.append(multiprocessing.Process(target=incrementar, args=( lock, shared)))

    start_time = time.perf_counter()
    # Se lanzan a ejecución
    for i, thread in enumerate(threads):
        thread.start()
        tiempos = random.randint(1, 3)
        print(f"Proceso {i+1} espera {tiempos} segundos.")
        time.sleep(tiempos)

    # y se espera a que todos terminen
    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
