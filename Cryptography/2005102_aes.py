import os
from BitVector import *
from timeit import default_timer as timer
import importlib

bitvectorDemo=importlib.import_module("2005102_bitvector_demo")


def calculateRcon():
    rcon=[0x00,0x01] 

    for i in range(2,11):  
        previousRcon=rcon[-1]

        if previousRcon<0x80:
            rcon.append(previousRcon*2)
        else:
            rcon.append(((previousRcon*2) ^ 0x11B) % 0x100) 
    
    return rcon


def substituteWord(word):
    #Substitute byte
    return [bitvectorDemo.Sbox[b] for b in word]

def rotateWord(word):
    #Left rotation
    return word[1:]+word[:1]

def xorWord(word1, word2):
    return [b1 ^ b2 for b1, b2 in zip(word1,word2)]

def keyExpansion(key,Rcon):
    w=[]
    for i in range(4):
        #Take 4 bytes of key in w0,w1,w2,w3
        w.append(list(key[4*i:4*(i+1)]))
        #print(w)
    for i in range(4,44):
        #Take 4 bytes of key in w4 to w43
        temp=w[i-1]
        if i%4 == 0:
            #Taking list of substitute word after rotation
            boxSubstituteWord=substituteWord(rotateWord(temp))
            # temp=g(w[3]) 
            temp=xorWord(boxSubstituteWord,[Rcon[i//4],0x00,0x00,0x00])
        w.append(xorWord(w[i-4],temp))
    #1-10 round keys
    roundKeys=[]
    for i in range(0,44,4):
        roundKey=w[i]+w[i+1]+w[i+2]+w[i+3]
        roundKeys.append(roundKey)
    return roundKeys

#Generalize
def keyExtendExpansion(key,Rcon,N4bytes,Nrounds): 
    w=[]
    for i in range(N4bytes):
        w.append(list(key[4*i:4*(i+1)]))
    for i in range(N4bytes,4*(Nrounds+1)): 
        temp=w[i-1]
        if i%N4bytes == 0:
            boxSubstituteWord=substituteWord(rotateWord(temp))
            temp=xorWord(boxSubstituteWord,[Rcon[i//N4bytes],0x00,0x00,0x00])

        #Additional condition for 256-bit
        elif N4bytes>6 and i%N4bytes == 4: 
            temp=substituteWord(temp)
        w.append(xorWord(w[i-N4bytes],temp))
    roundKeys=[]
    for i in range(0,len(w),4):
        roundKey=w[i]+w[i+1]+w[i+2]+w[i+3]
        roundKeys.append(roundKey)
    return roundKeys



def printRoundKeys(roundKeys):
    for idx, key in enumerate(roundKeys):
        print(f"Round {idx}: ",end="")
        for b in key:
            print(f"{b:02X}",end=" ")
        print("\n")


def createStateMatrix(block):
    # Create empty state
    state=[[0]*4 for _ in range(4)]

    # Convert block in 2D matrix
    for i in range(16):
        state[i%4][i//4]=block[i]
    return state



def addRoundKey(state,roundKey):
    for col in range(4):
        for row in range(4):
            # roundKey is in 1D block so 4*col+row gives actual key-value
            state[row][col] ^= roundKey[4*col+row]

def subBytes(state):
    for row in range(4):
        for col in range(4):
            state[row][col]=bitvectorDemo.Sbox[state[row][col]]

def shiftRows(state):
    for row in range(1,4):
        state[row]=state[row][row:]+state[row][:row]

def mixColumns(state):
    for col in range(4):
        tempCol=[BitVector(intVal=state[row][col],size=8) for row in range(4)]

        # To store multiplication value
        newCol=[0]*4
        for row in range(4):
            val=BitVector(intVal=0,size=8)
            for k in range(4):
                val ^= bitvectorDemo.Mixer[row][k].gf_multiply_modular(tempCol[k],bitvectorDemo.AES_modulus,8)
            newCol[row]=int(val)
        for row in range(4):
            state[row][col]=newCol[row]

# Padding according to PKCS#7
def pad(plaintext):
    padLen=16-(len(plaintext)%16)
    return plaintext+bytes([padLen]*padLen)

def unpad(plaintext):
    # Get the last byte value
    padLen=plaintext[-1]
    #Remove bytes from plaintext length of padLen
    return plaintext[:-padLen]


def printState(state,label):
    print(f"\n{label}")
    for row in state:
        #Convert each byte to 2 digit hexadecimal
        print(" ".join(f"{byte:02X}" for byte in row))


#AES Encryption
def aesEncryptCBC(plaintext,key,roundKeys):


    IV=os.urandom(16)
    ciphertext=list(IV)



    prevBlock=IV
    for blkStartIndex in range(0,len(plaintext),16):

        #Take 16 bytes
        block=list(plaintext[blkStartIndex:blkStartIndex+16])

        #XOR with IV/previous cipher
        block=[b^p for b,p in zip(block,prevBlock)]

        state=createStateMatrix(block)
        # printState(state,f"Initial State (Block {blkStartIndex//16})")

        addRoundKey(state,roundKeys[0])
        # printState(state, "After AddRoundKey (Round 0)")

        for rnd in range(1,10):
            subBytes(state)
            shiftRows(state)
            mixColumns(state)
            addRoundKey(state,roundKeys[rnd])
            # printState(state, f"After Round {rnd}")

        subBytes(state)
        shiftRows(state)
        addRoundKey(state,roundKeys[10])
        # printState(state, "After Final Round")



        cipherBlock=[]
        for col in range(4):
            for row in range(4):
                cipherBlock.append(state[row][col])

        for byte in cipherBlock:
            ciphertext.append(byte)

        prevBlock=cipherBlock


    return bytes(ciphertext)



def aesEncryptExtendCBC(plaintext,key,roundKeys,Nrounds):


    IV=os.urandom(16)
    ciphertext=list(IV)



    prevBlock=IV
    for blkStartIndex in range(0,len(plaintext),16):

        #Take 16 bytes
        block=list(plaintext[blkStartIndex:blkStartIndex+16])

        #XOR with IV/previous cipher
        block=[b^p for b,p in zip(block,prevBlock)]

        state=createStateMatrix(block)
        # printState(state,f"Initial State (Block {blkStartIndex//16})")

        addRoundKey(state,roundKeys[0])
        # printState(state, "After AddRoundKey (Round 0)")

        for rnd in range(1,Nrounds):
            subBytes(state)
            shiftRows(state)
            mixColumns(state)
            addRoundKey(state,roundKeys[rnd])
            # printState(state, f"After Round {rnd}")

        subBytes(state)
        shiftRows(state)
        addRoundKey(state,roundKeys[Nrounds])
        # printState(state, "After Final Round")



        cipherBlock=[]
        for col in range(4):
            for row in range(4):
                cipherBlock.append(state[row][col])

        for byte in cipherBlock:
            ciphertext.append(byte)

        prevBlock=cipherBlock


    return bytes(ciphertext)




def aesDecryptCBC(ciphertext,key,roundKeys):
    # roundKeys=keyExpansion(key)

    IV=list(ciphertext[:16])
    ciphertext=ciphertext[16:]

    plaintext=[]

    prevBlock=IV
    for blkStartIndex in range(0,len(ciphertext),16):
        cipherBlock=list(ciphertext[blkStartIndex:blkStartIndex+16])

        state=createStateMatrix(cipherBlock)
        addRoundKey(state,roundKeys[10])

        for rnd in range(9,0,-1):
            shiftRowsInverse(state)
            subBytesInverse(state)
            addRoundKey(state,roundKeys[rnd])
            mixColumnsInverse(state)

        shiftRowsInverse(state)
        subBytesInverse(state)
        addRoundKey(state,roundKeys[0])



        for col in range(4):
            for row in range(4):
                index=4*col+row 
                xorByte=state[row][col]^prevBlock[index]
                plaintext.append(xorByte)

        prevBlock=cipherBlock

    return bytes(plaintext)


def aesDecryptExtendCBC(ciphertext,key,roundKeys,Nrounds):
    # roundKeys=keyExpansion(key)

    IV=list(ciphertext[:16])
    ciphertext=ciphertext[16:]

    plaintext=[]

    prevBlock=IV
    for blkStartIndex in range(0,len(ciphertext),16):
        cipherBlock=list(ciphertext[blkStartIndex:blkStartIndex+16])

        state=createStateMatrix(cipherBlock)
        addRoundKey(state,roundKeys[Nrounds])

        for rnd in range(Nrounds-1,0,-1):
            shiftRowsInverse(state)
            subBytesInverse(state)
            addRoundKey(state,roundKeys[rnd])
            mixColumnsInverse(state)

        shiftRowsInverse(state)
        subBytesInverse(state)
        addRoundKey(state,roundKeys[0])



        for col in range(4):
            for row in range(4):
                index=4*col+row 
                xorByte=state[row][col]^prevBlock[index]
                plaintext.append(xorByte)

        prevBlock=cipherBlock

    return bytes(plaintext)




# SubBytes Inverse(for Decryption)
# InvSbox=[0]*256
# for i in range(256):
#     InvSbox[Sbox[i]]=i


def subBytesInverse(state):
    for row in range(4):
        for col in range(4):
            state[row][col]=bitvectorDemo.InvSbox[state[row][col]]

# ShiftRows Inverse (for Decryption)
def shiftRowsInverse(state):
    for row in range(1,4):
        # Shift Right
        state[row]=state[row][-row:]+state[row][:-row]

# InvMixer = [
#     [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
#     [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
#     [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
#     [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
# ]

def mixColumnsInverse(state):
    for col in range(4):
        tempCol=[BitVector(intVal=state[row][col],size=8) for row in range(4)]
        newCol=[0]*4
        for row in range(4):
            val=BitVector(intVal=0,size=8)
            for k in range(4):
                val ^= bitvectorDemo.InvMixer[row][k].gf_multiply_modular(tempCol[k],bitvectorDemo.AES_modulus,8)
            newCol[row]=int(val)
        for row in range(4):
            state[row][col]=newCol[row]



if __name__ == "__main__":

    plaintext=input("Enter plaintext: ")

    plaintext=plaintext.encode('utf-8')

    keyInput=input("Enter 16-character ASCII key: ")



    # Handle key size issues
    # 16bytes=128bits
    if len(keyInput)<16:
        keyInput=keyInput.ljust(16,' ')
    elif len(keyInput)>16:
        keyInput=keyInput[:16]

    
    # if len(keyInput)==16:
    #     pass 
    # elif len(keyInput)==24:
    #     pass 
    # elif len(keyInput)==32:
    #     pass 
    # elif len(keyInput)<16:
    #     keyInput=keyInput.ljust(16,' ')
    # elif 16<len(keyInput)<24:
    #     keyInput = keyInput.ljust(24,' ')
    # elif 24<len(keyInput)<32:
    #     keyInput=keyInput.ljust(32,' ')
    # else:
    #     keyInput=keyInput[:32]


    key=keyInput.encode('utf-8')

    keyLen=len(key)
    if keyLen==16:
        N4bytes,Nrounds=4,10
    elif keyLen==24:
        N4bytes,Nrounds=6,12
    elif keyLen==32:
        N4bytes,Nrounds=8,14

    # print(N4bytes,Nrounds)
      
    startTime=timer()  
    Rcon=calculateRcon()


    roundKeys=keyExpansion(key,Rcon)
    # roundKeys=keyExtendExpansion(key,Rcon,N4bytes,Nrounds)


    schedulingTime=timer()-startTime
    schedulingTime*=1000

    
    # printRoundKeys(roundKeys)

    print("\nKey:")
    print(f"In ASCII: {keyInput}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in key)}")

    print("\nPlain Text:")
    print(f"In ASCII: {plaintext.decode('utf-8')}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in plaintext)}")


    paddedPlaintext=pad(plaintext)
    print(f"In ASCII (After Padding): {paddedPlaintext.decode('utf-8')}")
    print(f"In HEX (After Padding): {' '.join(f'{b:02x}' for b in paddedPlaintext)}")

    print("\nCiphered Text:")
    startTime=timer()


    ciphertext=aesEncryptCBC(paddedPlaintext,key,roundKeys)
    # ciphertext=aesEncryptExtendCBC(paddedPlaintext,key,roundKeys,Nrounds)


    encryptionTime=timer()-startTime
    encryptionTime*=1000
    print(f"In HEX: {' '.join(f'{b:02x}' for b in ciphertext)}")

    print(f"In ASCII: {''.join([chr(byte) for byte in ciphertext])}")

    print("\nDeciphered Text:")
    startTime=timer()
    print("Before Unpadding:")


    decryptedTextBefore=aesDecryptCBC(ciphertext,key,roundKeys)
    # decryptedTextBefore=aesDecryptExtendCBC(ciphertext,key,roundKeys,Nrounds)


    decryptionTime=timer()-startTime
    decryptionTime*=1000
    print(f"In HEX: {' '.join(f'{b:02x}' for b in decryptedTextBefore)}")
    print(f"In ASCII: {decryptedTextBefore.decode('utf-8')}")


    print("After Unpadding:")
    decryptedTextAfter=unpad(decryptedTextBefore)
    print(f"In ASCII: {decryptedTextAfter.decode('utf-8')}")
    print(f"In HEX: {' '.join(f'{b:02x}' for b in decryptedTextAfter)}")

    print("\nExecution Time Details:")
    print(f"Key Schedule Time: {schedulingTime:.6f} ms")
    print(f"Encryption Time: {encryptionTime:.6f} ms")
    print(f"Decryption Time: {decryptionTime:.6f} ms")