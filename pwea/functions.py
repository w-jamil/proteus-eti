def subs(weights,pred,eta, minY, maxY):
    gmin = -(1/eta) * numpy.log(numpy.dot(weights, (numpy.exp(-eta * (pred - minY)**2))))
    gmax = -(1/eta) * numpy.log(numpy.dot(weights, (numpy.exp(-eta * (pred - maxY)**2))))
    g = (0.5 * (minY + maxY)) - (gmax - gmin)/(2 *(maxY - minY))
    return g

def AA(experts,actual,eta, minY, maxY):
    n = numpy.size(experts, axis=1)
    t = numpy.size(experts, axis=0)
    predictions = numpy.zeros(t)
    w = numpy.ones(n)
    for i in range(0,t):
        w = w/w.sum()
        predictions[i] = subs(w,experts[i,],eta, minY, maxY)
        w = w * numpy.exp(-eta * (experts[i,]-actual[i])**2)
    return predictions

###################Ensambling strategy WAA

def WAA(experts,actual,eta):
    n = numpy.size(experts, axis=1)
    t = numpy.size(experts, axis=0)
    predictions = numpy.zeros(t)
    w = numpy.ones(n)
    for i in range(0,t):
        w = w/w.sum()
        predictions[i] = numpy.dot(w,experts[i,])
        w = w * numpy.exp(-1.0/eta * (experts[i,]-actual[i])**2)
    return predictions

#############################switching
    
def SEAA(experts,actual,eta, minY, maxY,alpha):
    n = numpy.size(experts, axis=1)
    t = numpy.size(experts, axis=0)
    predictions = numpy.zeros(t)
    w = numpy.ones(n)
    for i in range(0,t):
        w = w/w.sum()
        w = ((1.0-alpha) * w) + ((alpha/(n-1.0)) * (1.0-w))
        predictions[i] = subs(w,experts[i,],eta, minY, maxY)
        w = w * numpy.exp(-eta * (experts[i,]-actual[i])**2)
    return predictions
