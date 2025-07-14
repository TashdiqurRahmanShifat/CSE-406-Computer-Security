import socket
import pickle
import random
import importlib


aes=importlib.import_module("2005102_aes")
aesEncryptCBC=aes.aesEncryptCBC
pad=aes.pad
calculateRcon=aes.calculateRcon

ecc=importlib.import_module("2005102_ecc")



curve,G,p,a,b=ecc.generateCurvePoints(128)

#Private key generation
Ka=random.getrandbits(128)

#Public key generation
A=curve.doubleAndAdd(Ka,G)
try:
    client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect(('localhost',1234))

    client.send("I want to send you a encrypted message".encode())
    startReply=client.recv((4096)).decode()
    if startReply!="y":
        print("Server does not agree to encrypt message.Client connection is closing...")
        client.close()
        exit()

    #Client sends a,b,G,p,A
    client.send(pickle.dumps({"a":a,"b":b,"p":p,"G":G,"A":A}))
    # print(f"Client is sending client's public key: {A}")

    #Client receives server's public key
    B=pickle.loads(client.recv(4096))
    # print(f"Client received server's public key: {B}")

    #Compute shared secret key
    R=curve.doubleAndAdd(Ka,B)
    sharedKey=R[0].to_bytes(16,byteorder='big')

    # print(sharedKey)

    #Waiting for server to get confirmation message
    # serverConfMsg=client.recv(4096).decode()
    # print(serverConfMsg)
    # if serverConfMsg != "Encryption Key Generated":
    #     print("Client did not receive READY signal from Server.Aborting...")
    #     client.close()
    #     exit()

    readyMsg=client.recv(4096).decode()
    print(f"[From Server]-{readyMsg}")

    response=input("Your response(y/n):")
    client.send(response.encode())
    if(response=='n'):
        print("Client is not ready!")
        client.close()
        exit()
    print("Key exchange confirmed.Proceeding to send message...")

    #Type message which is going to be sent
    msg=input("Enter message to encrypt and send: ").encode()

    #Calculate round constants
    Rcon=calculateRcon()

    roundKeys=aes.keyExpansion(sharedKey,Rcon)
    msg=pad(msg)
    ciphertext=aesEncryptCBC(msg, sharedKey, roundKeys)

    # print(Rcon)
    # print(f"In ASCII: {''.join([chr(byte) for byte in ciphertext])}")

    #Send encrypted data
    client.send(pickle.dumps({"cipher": ciphertext,"rcon": Rcon}))
    print("Message sent!")
    client.close()

except:
    print("Connection closed")