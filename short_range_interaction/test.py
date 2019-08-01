L = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
result = []
for i in range(len(L)-12+1):
    temp_min = float('inf') 
    for j in range(12):
        temp_min = min(temp_min,L[i+j])
    result.append(temp_min)
print(result)