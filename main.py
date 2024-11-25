
# this program that get all prime numbers in the range
from typing import List

from fastapi import FastAPI
def find_primes(x,y):
    primes=[]

    for num in range (x,y+1):
        if num>1:
            for i in range (2,num):
                if(num%i==0):

                    break
            else:
                primes.append(num)

    return primes



def convertToBinary(num):
    binary=[]
    while num!=0:
        remainder=num%2
        binary.append(remainder)
        num=num//2

    binary=binary[::-1]
    st=""
    for i in binary:
        st+=str(i)


    return st


def count_vowles(string):
    count=0
    string=string.lower()
    for i in string:
        if i in "aeiou":
            count+=1

    return count

def bubble_sort(arr: List[int]) -> List[int]:
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

app = FastAPI()

@app.get("/primes/{x}/{y}")
def read_item(x: int, y: int):
    return find_primes(x,y)

@app.get("/binary/{num}")
def read_item(num: int):
    return convertToBinary(num)

@app.get("/vowels/{string}")
def read_item(string: str):
    return count_vowles(string)

@app.post("/primes/{x}/{y}")
def read_item(x: int, y: int):
    print(find_primes(x,y))


@app.get("/sort/{arr}")
async def sort_array(arr: str) -> dict:
    try:
        arr_list = [int(x) for x in arr.split(',')]
        sorted_list = bubble_sort(arr_list)
        return {"sorted_array": sorted_list}
    except ValueError:
        return {"error": "Invalid input. Please provide comma-separated integers"}
