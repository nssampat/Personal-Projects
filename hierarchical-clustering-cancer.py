'''
Background Information:

Data Set
This project analyzes the Breast Cancer Wisconsin Diagnostic Data Set. Each of the 569 data points in this set 
represents a tumor from a particular patient, and each data point has 32 attributes (2 of which are the ID and 
the classification as Benign, 'B', or Malignant, 'M'). Other attributes include radius, texture, perimeter,
area, etc. 

Algorithm
Hierarchical clustering with complete linkage was the algorithm used on this dataset. This clustering algorithm
has two variations - single linkage and complete linkage. With single linkage, the distance between each point 
and every other point is calculated. Then, we sort the distances from smallest to largest and begin clustering 
the points using the following rules. Keep in mind that we grab points based on the smallest distances and keep 
going until we form one big cluster. To start off, the points which are closest to each other are put into a 
cluster.

Single-Linkage
1. If neither point is currently in a cluster, a new cluster is formed with the two points.
2. If one point is in a cluster, but the other isn't, the other point is added to the same cluster as the original
point.
3. If both points are clustered already, but belong to different clusters, the two clusters are merged.
4. If both points are already in the same cluster, we do nothing and move to the next pair of points.

However, single-linkage often fails to accurately cluster data. These are discussed below.

Challenges
After implementing a single-linkage hierarchical clustering algorithm on this dataset, two clusters were formed,
but they were not very accurate - one cluster contained just 5 points, and the other cluster contained the other
564 data points. At first, I thought some or all of these 5 points were outliers, so they were removed from the
dataset in various combinations. However, the problem persisted, so I turned to complete linkage as a solution.
Instead of calculating the distance between points, complete-linkage instead calculates the distances between 
clusters, where the distance between clusters is defined as the farthest distance between two points (one 
from each of the clusters). This is a better approach since clusters are now merged based on their least-
similar elements, rather than their most-similar elements - before, if one point in a cluster was really
similar to a point in a different cluster, the clusters would be merged even if the majority of the points in
the clusters were not very similar. 

Another problem I faced was the different scales of each parameter in the data set. This was resolved by 
iterating through the entire data file and normalizing each column of data.

Finally, I wanted to reduce the runtime of the clustering algorithm, so I decided not to sort the distances 
list and instead write functions called get_index and get_distance which would allow me to traverse the 
unsorted distances list and grab the correct distance based on the two IDs we want to find the distance between.
Once these distances were grabbed, they were put into a smaller list and sorted using the get_index and 
get_distance functions. This cut down the runtime by about 33%, since we were no longer iterating through all of
distances (which has a size of 161596) in order to find just one distance.

Future Work
A Principal Component Analysis could be performed on this data set before implementation of the algorithm,
and the results compared to that of this clustering algorithm without PCA. PCA was not implemented on this 
dataset because all of the parameters had quite a bit of variation across different data points (unlike the
MNIST data set, for example, in which many of the pixel intensities were 0).

The algorithm could also be run directly on images of tumors and other medical images, rather than using 
measurements extracted from those images which could be inaccurate. However, a method of standardizing the size
and other features of the images would have to be found.

Sources
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
https://www.displayr.com/strengths-weaknesses-hierarchical-clustering/

'''


import csv
import math
import statistics

filename = "tumor.csv" #The first two columns are the tumor ID and its classification. The other 30 columns are data.
file = open(filename, 'r')

for i, row in enumerate(file):
    pass
num_points = i + 1 #There are 569 points in this data set.

k = 2 #For this data set, we only want 2 clusters - benign and malignant tumors.
attributes = 32 #Each data point has 32 attributes.
coord_length = attributes - 2 #We don't want the ID number or classification.

malignant = 0 #This will help record the true number of malignant tumors in the dataset.
benign = 0 #This will help record the true number of benign tumors in the dataset.
malignant_ids = [] #This list will record the IDs of all malignant tumors.
benign_ids = [] #This list will record the IDs of all benign tumors.


file.seek(0)
new_rows = [row.split(",") for row in file.readlines()] #This stores all of the row data in a list.

for x in range(len(new_rows)):
    new_rows[x][-1] = new_rows[x][-1].replace("\n", "") #Here, we remove the newline character from the last data point in each row.

#Normalization of the Data
'''Right now, not all of the measurements are on the same scale. For example, one parameter has values between 0
and 1, while another parameter has values in the thousands. We need to make sure all of the data is on the same
scale, so we normalize the data points as follows.'''

#Storing the Columns
updated_columns = [] #This is to keep track of all of the updated (normalized) values for the columns.
for i in range(attributes): #We will need to do this for each of the 32 columns.
    listx = [] #This is the temporary list containing the values for one given column.
    for x in range(len(new_rows)):
        if i < 2:
            listx.append(new_rows[x][i]) #The first two columns are just the ID and classification, so we don't need to convert these to floats.
        else:
            listx.append(float(new_rows[x][i])) #We append the ith element of the row for each row - this is just the same as all of the elements in a column.
    updated_columns.append(listx) #Finally, we append all of the columns to updated columns - keep in mind we have not updated them yet!

#Updating the Columns
for i in range(len(updated_columns)):
    if i < 2:
        pass #The first two columns are just the ID and classification, so we don't need to normalize these.
    else:
        minimum = min(updated_columns[i]) #We find the minimum for a given column.
        maximum = max(updated_columns[i]) #We find the maximum for a given column.
        for x in range(len(updated_columns[i])): 
            updated_columns[i][x] = (updated_columns[i][x] - minimum)/(maximum-minimum) #We normalize by subtracting the column minimum from each data point, then dividing by (maximum - minimum). This will cause the lowest value to be 0 and the highest to be 1. 


file.close() #We close the original file here.

#Writing the Updated Values into a New File.
new_filename = 'tumor_new.csv' #We create a new file to store the normalized data.
new_file = open(new_filename, 'w')
writer = csv.writer(new_file)

for i in range(num_points):
    writer.writerow([updated_columns[x][i] for x in range(len(updated_columns))]) #The length of updated_columns is 32. We iterate through each column in updated_columns and add the ith element (using a list comprehension, we add a full row at a time.)

new_file.close() #We close the new file.
filename = new_filename #We redefine filename to be the new file's name.
file = open(filename, 'r') #We reopen the new file, but under the original name file.

file.seek(0)
reader = csv.reader(file)

ids = [None] * num_points #Generates a list of length num_points with Nones.
id_dictionary = {} #This will store the ID number and row number of a point in a dictionary.
for i, row in enumerate(reader):
    ids[i] = row[0] #Puts all of the IDs in a list called ids.
    id_dictionary[row[0]] = i #Puts all of the IDs and the row numbers in the id_dictionary.
    if row[1] == "M":
        malignant += 1 #Counts the number of malignant tumors.
        malignant_ids.append(row[0]) #Records the IDs of malignant tumors.
    elif row[1] == "B":
        benign += 1 #Counts the number of benign tumors.
        benign_ids.append(row[0]) #Records the IDs of benign tumors.
        
file.seek(0) #Seek back to beginning of file.

print("Here is some useful information about the dataset:")
print("Number of data points: ", num_points)
print("Number of attributes per data point (excluding ID number and classification): ", coord_length)
print("The number of malignant tumors in the data set is " + str(malignant) + ", and the number of benign tumors in the data set is " + str(benign) + ".")

print(coord_length)
print(id_dictionary) #Print the ID dictionary to make sure they were recorded properly.

def distance(point1, point2):
    summation = 0
    for i in range(coord_length):
        summation += (float(point1[i]) - float(point2[i]))**2 #Calculates square of Euclidean distance by comparing each coordinate of the first point to that of the second point.    
    dist = math.sqrt(summation) #Take the square root.
    return dist #Return the distance.
  
file.seek(0) #Seek back to beginning of file before performing calculations.
distances = [] #This list will contain the distances between each point and every other point.

#For each point, calculate the Euclidean distance between the point and every other point.
#We will use readlines instead of our reader here so we don't have to worry about seeking to the correct line.
rows = [[i, row.split(",")] for i, row in enumerate(file.readlines())] #This stores all of the row data in a list along with the row number.
for x in range(len(rows)):
    rows[x][1][-1] = rows[x][1][-1].replace("\n", "") #Here, we remove the newline character from the last data point in each row.


for i in range(len(rows)):
    point_id = rows[i][1][0] #Take the point ID.
    point = rows[i][1][2:] #Take the point's data, excluding its ID and classification as benign or malignant.
    for x in range(i+1, len(rows)): #Only calculate the distance between the point and points which come after it in the file.
        other_point_id = rows[x][1][0] #Take the other point's ID.
        other_point = rows[x][1][2:] #Take the other point's data, excluding its ID and classification as benign or malignant.
        distances.append([distance(point, other_point), point_id, other_point_id]) #Append the distance, along with the two IDs of the points we used to calculate the distance, to the distances list.

unsorted_distances = distances[:] #We actually want to keep the distances unsorted so that we can pinpoint the distance between two IDs more easily.
print("Unsorted Distances: ", unsorted_distances[57632]) #Print some element from unsorted distances.
print(len(distances)) #Make sure the length of distances is equal to num_points * (num_points - 1)/2.


def get_index(idvalue1, idvalue2): #Given two point IDs, returns the index of the corresponding distance in the unsorted_distances list.
    idindex1 = id_dictionary[idvalue1] #Grab the row number of the first point.
    idindex2 = id_dictionary[idvalue2] #Grab the row number of the second point.
    if idindex1 > idindex2:
        idindex1, idindex2 = idindex2, idindex1 #Make sure that idindex1 (row number) is the smaller one.
    if idindex1 != 0:
        first_value = num_points - idindex1
        summation = 0
        for number in range(first_value, num_points):
            summation += number #Figure out how many distances have been calculated for row numbers before this one.
    else: 
        summation = 0 #If the first row number is zero, we set summation equal to zero.
    second_value = idindex2 - idindex1 #Figure out how many distances have been calculated for this row number already.
    return summation + second_value - 1 #Add them to figure out the final index of the distance between these two points in the unsorted_distances list.

'''Uncomment the below print statement to test the function.'''
# print(get_index('86409', '898677'))

def get_distance(cluster1, cluster2): #Given two clusters, find the maximum distance between any two points (one from each cluster). This will serve as the 'distance' between two clusters.
    sub_distances = [] #This is where all of the distances between points in the two clusters will go.
    for id_value1 in cluster1:
        for id_value2 in cluster2:
            element = unsorted_distances[get_index(id_value1, id_value2)] #Given the IDs of two points, we grab the index of the corresponding distance using the get_index function. Then, we use this index in unsorted_distances to grab the actual distance (as well as the IDs).
            sub_distances.append(element) #We append this selected distance to sub_distances.
    sub_distances.sort() #We sort sub_distances.
    return sub_distances[-1] #We take the maximum distance.
  
'''Complete linkage involves taking the distances between clusters themselves (using a farthest neighbor 
technique), rather than between points. This was implemented after single linkage failed.'''

clusters = [[id_value] for id_value in ids] #Clusters is initialized to be a list of lists, each containing one ID.
while len(clusters) > k: #We want to keep clustering as long as there are more than 2 clusters.
    iterations = int(len(clusters)*(len(clusters) - 1)/2) #This is the total number of distances we will have to get for a given round of clustering.
    mega_distances = [0]*iterations #Initialize mega_distances, which will store all of the distances between clusters, to have a length equal to iterations.
    counter = 0
    for i in range(len(clusters)):
        for x in range(i+1, len(clusters)):
            mega_distances[counter] = get_distance(clusters[i], clusters[x]) #get_distance will return the max distance and IDs between two clusters. We then put this into the mega_distances list.
            counter += 1
    mega_distances.sort() #We sort mega_distances.
    closest = mega_distances[0] #We take the closest 2 clusters after sorting.
    id1 = closest[1] #Take the ID of the 'farthest neighbor point' in the closest cluster pair.
    id2 = closest[2] #Take the ID of the other 'farthest neighbor point' in the closest cluster pair.
    id1_index = None
    id2_index = None
    for cluster in clusters:
        if id1 in cluster:
            id1_index = clusters.index(cluster) #Take the index of the cluster which contains the relevant farthest neighbor point.
        if id2 in cluster:
            id2_index = clusters.index(cluster) #Take the index of the cluster which contains the other relevant farthest neighbor point.
    clusters[id1_index].extend(clusters[id2_index]) #Add all of the IDs in the second cluster to the first cluster.
    clusters[id2_index] = [] #Set the second cluster equal to an empty list.
    clusters = [item for item in clusters if item != []] #Remove all empty lists from the clusters list.
    print(len(clusters)) #Print the length of clusters so we can see how much more clustering remains.

print("Clusters: ", clusters)

for cluster in clusters:
    print(cluster, len(cluster)) #Prints out how many points are in each cluster.
    
classifications = [] 

for cluster in clusters:
    classifications.append(cluster[:]) #Make a copy of the final clusters called classifications.

for element in classifications:
    for i in range(len(element)):
        if element[i] in malignant_ids:
            element[i] = 'M' #Replace all IDs of malignant tumors with an M.
        else:
            element[i] = 'B' #Replace all IDs of benign tumors with a B.
            
print("Classifications: ", classifications) #Print the classifications so we can see how well our clustering worked.
print("Clusters: ", clusters)

def purity():
    purity_score = 0
    for cluster in classifications:
        mode = max(statistics.multimode(cluster)) #For each cluster, we find the mode.
        purity_score += cluster.count(mode) #We add the number of occurrences of the mode to purity score.
    purity_score = purity_score/num_points #Divide purity score sum by the number of total points.
    return purity_score

print("Purity: ", purity())

def silhouette(i):
    for cluster in clusters:
        if ids[i] in cluster:
            cluster_index = clusters.index(cluster) #Grab the index of the cluster containing the specific point ID in the clusters list.
        if len(cluster) == 1:
            return 0 #Returns 0 if there is only one point in the cluster.
    distances_silhouette = [0] * k #This list of distances will include a_i as well as b_i.
    point_id = ids[i] #The ID of the chosen point is ids at i.
    for other_id in ids: #Iterate through all of the other ids.
        for cluster in clusters:
            if other_id in cluster:
                other_cluster_index = clusters.index(cluster) #Grab the index of the cluster containing the other point's ID in the clusters list.
        if other_id != point_id: #If the points are not the same, proceed.
            distances_silhouette[other_cluster_index] += float(unsorted_distances[get_index(point_id, other_id)][0]) #Given the IDs of two points, we grab the index of the corresponding distance using the get_index function. Then, we use this index in unsorted_distances to grab the actual distance, which we add to the corresponding silhouette distance.
    for cluster in clusters:
        if clusters.index(cluster) == cluster_index:
            count = len(cluster) - 1 #If the cluster is the one that the point is in, subtract one from the length of the cluster (since we want the number of distances in the list).
        else:
            count = len(cluster)
        distances_silhouette[clusters.index(cluster)] /= count #Divide the silhouette distances by the count in the cluster.
    if cluster_index == 0:
        other_cluster_index = 1
    elif cluster_index == 1:
        other_cluster_index = 0
    a = distances_silhouette[cluster_index] #Selects the average distance of the chosen point to points in its own cluster.
    b = distances_silhouette[other_cluster_index] #Selects the lowest of the average distances of the chosen point to points in other clusters.
    return (b-a)/max(a,b) #Returns the silhouette value for that point.

avg_silhouette = sum([silhouette(i) for i in range(num_points)])/num_points #Calculate average silhouette value.
print("Average Silhouette Value: ", avg_silhouette)
