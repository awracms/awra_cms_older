
def test_fanin():
    from time import sleep
    from multiprocessing import Queue as MPQueue
    from awrams.utils.messaging.brokers import FanInChunkBroker
    from awrams.utils.messaging.general import message

    NWORKERS = 4

    m2b,b2m,outq = MPQueue(),MPQueue(),MPQueue(1)

    qin = dict(control=m2b)
    for i in range(NWORKERS):
        qin[i] = MPQueue()

    qout = dict(control=b2m,out=outq,workers=MPQueue())

    broker = FanInChunkBroker(qin,qout,NWORKERS)
    broker.start()

    sleep(1)
    # Send some input
    i = 0
    for x in range(100):
        #sleep(0.0001)
        print(x)
        qin['control'].put(message(i))

        for y in range(4):
            print(y)
            # outpipes[i%16].send(i)
            qin[y].put(message(i))
            print("i",i)
            i += 1
            print("WORKERS",qout['workers'].get())
            print("OUTQ",outq.get())


    #for x in range(1600*16):
    #    outq.get()

    m2b.put(message('terminate'))
    broker.join()

    resp = b2m.get()
    print(resp)
    assert resp['content']['obj_class'] == 'FanInChunkBroker'
    assert resp['subject'] == 'terminated'

if __name__ == '__main__':
    test_fanin()
