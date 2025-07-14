import random
from timeit import default_timer as timer
from sympy import nextprime

#Euler's Criterion(Check square root exists)
def eulerCriterion(n,p):
    return pow(n,(p-1)//2,p) == 1

#Tonelli-Shanks(Compute square root)
def tonelliShanks(n,p):
    if not eulerCriterion(n,p):
        raise ValueError("n is not a quadratic residue")

    if p%4 == 3:
        return pow(n,(p+1)//4,p)

    #Step 1
    Q,S=p-1,0
    while Q%2 == 0:
        Q//=2
        #Q is divided by 2 each time,so s will increase by 1 each time as p-1=Q2^S
        S+=1

    #Step 2
    for z in range(2,p):
        if not eulerCriterion(z,p):
            break


    #Step 3
    M=S
    c=pow(z,Q,p)
    t=pow(n,Q,p)
    R=pow(n,(Q+1)//2,p)
    
    #Step 4
    while t!=1:
        #t!=1
        i,temp=0,t
        #Repeated Squaring until t^(2^i)
        while temp!=1:
            temp=pow(temp,2,p)
            i+=1
        b=pow(c,2**(M-i-1),p)
        R=(R*b)%p
        t=(t*b*b)%p
        c=(b*b)%p
        M=i
    return R


#Generate prime
def generatePrime(bits):
    p=random.getrandbits(bits)
    return nextprime(p)


#Generate Curve and paramter a,b,G
def generateCurvePoints(bits):
    random.seed(0)
    while True:
        p=generatePrime(bits)
        a=random.randrange(0,p)
        b=random.randrange(0,p)
        try:
            curve=EllipticCurve(a,b,p)
        except ValueError:
            continue

        for _ in range(1000):

            #Randomly take x
            x=random.randrange(0,p)

            #y^2=x^3+ax+b mod p
            rhs=(x**3+a*x+b)%p

            if eulerCriterion(rhs,p): 
                try:
                    y=tonelliShanks(rhs,p)
                    if not curve.isOnCurve(x,y):
                        continue
                    G=(x,y)
                    return curve,G,p,a,b
                except AssertionError:
                    continue


class EllipticCurve:

    def __init__(self,a,b,p):

        #Singularity Check
        if (4*a**3+27*b**2)%p==0:
            raise ValueError("Singular curve")
        
        self.a=a
        self.b=b
        self.p=p

    def isOnCurve(self,x,y):
        return (y**2-(x**3+self.a*x+self.b))%self.p == 0

    def pointAddition(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        
        x1,y1=P
        x2,y2=Q

        #Point at infinity
        if x1 == x2 and y1 != y2:
            return None  

        #Point Doubling
        if P == Q:
            s=(3*x1**2+self.a)*pow(2*y1,-1,self.p)


        #Point Addition
        else:
            s=(y2-y1)*pow(x2-x1,-1,self.p)

        s%=self.p
        x3=(s**2-x1-x2)%self.p
        y3=(s*(x1-x3)-y1)%self.p
        return (x3,y3)

    #Point Multiplication
    def doubleAndAdd(self,k,P):
        #13=1101=>P;(2P+P);2(2P+P);((2(2P+P)))2+P
        result=None
        point=P
        for bit in bin(k)[2:]:
            result=self.pointAddition(result,result) if result else None
            if bit == '1':
                result=self.pointAddition(result,point) if result else point
        return result


avgTime1=[0]*3
avgTime2=[0]*3
avgTime3=[0]*3


def mainProgram(bitSize,trials):

    totalTimeA=0
    totalTimeB=0
    totalTimeR=0
    for _ in range(trials):
        curve,G,p,a,b=generateCurvePoints(bitSize)

        Ka=random.getrandbits(bitSize)
        Kb=random.getrandbits(bitSize)

        #Alice's public key=A
        start=timer()
        A=curve.doubleAndAdd(Ka,G)
        timeA=timer()-start

        #Bob's public key=B
        start=timer()
        B=curve.doubleAndAdd(Kb,G)
        timeB=timer()-start

        #Shared Key
        start=timer()
        R1=curve.doubleAndAdd(Ka,B)
        R2=curve.doubleAndAdd(Kb,A)
        timeR=timer()-start

        # if R1 == R2:
        #     print("yes")
        # else: print("no")
        shared_key = R1[0]

        totalTimeA+=timeA
        totalTimeB+=timeB
        totalTimeR+=timeR


    #     print(f"\nTrial:")
    #     print(f"P = {p}")
    #     print(f"a = {a}")
    #     print(f"b = {b}")
    #     print(f"G = {G}")
    #     print(f"Ka = {Ka}, A = {A}")
    #     print(f"Kb = {Kb}, B = {B}")
    #     print(f"Shared Key R = {shared_key}")

    # print(f"\nECDH Timing for {bitSize}-bit security over {trials} trials:")
    # print(f"{'Computation':<15}Time (s)")
    # print(f"{'-'*30}")
    # print(f"{'Alice (Ka*G)':<15}{totalTimeA/trials:.6f}")
    # print(f"{'Bob (Kb*G)':<15}{totalTimeB/trials:.6f}")
    # print(f"{'Shared Key':<15}{totalTimeR/trials:.6f}")

    if bitSize==128:
        avgTime1[0]=(totalTimeA*1000)/trials
        avgTime1[1]=(totalTimeB*1000)/trials
        avgTime1[2]=(totalTimeR*1000)/trials
    elif bitSize==192:
        avgTime2[0]=(totalTimeA*1000)/trials
        avgTime2[1]=(totalTimeB*1000)/trials
        avgTime2[2]=(totalTimeR*1000)/trials
    elif bitSize==256:
        avgTime3[0]=(totalTimeA*1000)/trials
        avgTime3[1]=(totalTimeB*1000)/trials
        avgTime3[2]=(totalTimeR*1000)/trials


def printTable():
    print("+" + "-"*6 + "+" + "-"*32 + "+")

    print("|      |  Computation Time For(in ms)   |")
    print("|  k   |--------------------------------|")
    print("|      |    A   |    B   | shared key R |")

    print("|------|--------|--------|--------------|")

    print(f"| {128:^4} | {round(avgTime1[0],2):^6.2f} | {round(avgTime1[1],2):^6.2f} | {round(avgTime1[2],2):^12.2f} |")
    print(f"| {192:^4} | {round(avgTime2[0],2):^6.2f} | {round(avgTime2[1],2):^6.2f} | {round(avgTime2[2],2):^12.2f} |")
    print(f"| {256:^4} | {round(avgTime3[0],2):^6.2f} | {round(avgTime3[1],2):^6.2f} | {round(avgTime3[2],2):^12.2f} |")

    print("+" + "-"*6 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*14 + "+")


if __name__ == "__main__":
    for bitSize in [128,192,256]:
        mainProgram(bitSize,trials=5)

    printTable()