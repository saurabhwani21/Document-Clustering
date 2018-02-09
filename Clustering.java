/**
 * Filename	   : Clustering.java
 * Modified by : Saurabh A. Wani 
 * Date        : 12/08/2017
 */

package lab5;

import java.util.*;

/**
 * Document clustering
 * 
 * @author qyuvks
 *
 */
public class Clustering {

	// String array that will contain data from all files.
	String[] myDocs;
	// List which contain all unique terms / Postings
	ArrayList<String> termList;
	// Termlists with postings
	ArrayList<ArrayList<Doc>> docLists;
	// 2D list to store the matrix
	ArrayList<ArrayList<Double>> matrix;
	// Total number of clusters to be generated.
	int numOfClusters;
	// List to store all the clusters.
	ArrayList<Cluster> listOfClusters;
	// Maximum iterations before stopping the K-means clustering.
	static final int totalIterations = 10;

	/**
	 * Constructor for attribute initialization
	 * 
	 * @param numC number of clusters
	 */
	public Clustering(int numC) {
		this.numOfClusters = numC;
		listOfClusters = new ArrayList<Cluster>();
	}

	/**
	 * Load the documents to build the vector representations
	 * 
	 * @param docs Documents to be processed. 
	 */
	public void preprocess(String[] docs) {
		myDocs = docs;
		termList = new ArrayList<String>();
		docLists = new ArrayList<ArrayList<Doc>>();
		ArrayList<Doc> docList;
		matrix = new ArrayList<ArrayList<Double>>(docs.length);
		for (int i = 0; i < myDocs.length; i++) {
			// Splitting contents of the file and generating tokens.
			String[] tokens = myDocs[i].split("[^a-zA-Z0-9']");
			for (String token : tokens) {
				if (token.length() <= 1)
					continue;
				// If token does not exist then add to token list
				if (!termList.contains(token)) {
					termList.add(token);
					docList = new ArrayList<Doc>();
					docList.add(new Doc(i, 1));
					docLists.add(docList);
				}
				// If the token exists.
				else {
					int index = termList.indexOf(token);
					docList = docLists.get(index);
					int temp = 0;
					for (Doc d : docList) {
						if (d.docID == i) {
							d.tf += 1;
							temp = 1;
							break;
						}
					}
					if (temp == 0)
						docList.add(new Doc(i, 1));
				}
			}
		}
		generateMatrix();
	}

	// Generates a document vector, where each vector would be the frequency of
	// every term.
	public void generateMatrix() {
		int counter = 0;
		while (counter < myDocs.length) {
			ArrayList<Double> document = new ArrayList<Double>(termList.size());
			for (int i = 0; i < termList.size(); i++)
				document.add(0.0);
			matrix.add(counter, document);
			counter += 1;
		}
		int index = 0;
		for (ArrayList<Doc> contents : docLists) {
			for (Doc doc : contents)
				matrix.get(doc.docID).set(index, 1 + Math.log(doc.tf));
			index += 1;
		}
	}

	/**
	 * Cluster the documents For kmeans clustering, use the first and the ninth
	 * documents as the initial centroids
	 */
	public void cluster() {
		for (int i = 0; i < numOfClusters; i++)
			listOfClusters.add(new Cluster(termList.size()));
		listOfClusters.get(0).centroid = matrix.get(0);
		listOfClusters.get(1).centroid = matrix.get(9);
		// Store number of iterations performed.
		int iterations = 0;
		// Store distance.
		double d1, d2;
		do {
			iterations += 1;
			for (int docID = 0; docID < matrix.size(); docID++) {
				ArrayList<Double> content = matrix.get(docID);
				d1 = listOfClusters.get(0).cosineSimilarity(content);
				d2 = listOfClusters.get(1).cosineSimilarity(content);
				if (d1 <= d2)
					listOfClusters.get(0).addContent(content, docID);
				else
					listOfClusters.get(1).addContent(content, docID);
			}
			listOfClusters.get(0).compCentroid();
			listOfClusters.get(1).compCentroid();
		} while (iterations <= totalIterations);
		// Print the clusters.
		int clusterNum = 0;
		for (int i = listOfClusters.size() - 1; i >= 0; i--)
			System.out.println("Cluster: " + clusterNum++ + "\n" + listOfClusters.get(i));
	}

	public static void main(String[] args) {
		String[] docs = { "hot chocolate cocoa beans", "cocoa ghana africa", "beans harvest ghana", "cocoa butter",
				"butter truffles", "sweet chocolate can", "brazil sweet sugar can", "suger can brazil",
				"sweet cake icing", "cake black forest" };
		Clustering c = new Clustering(2);

		c.preprocess(docs);

		c.cluster();

		/*
		 * Expected result: 
		 * Cluster: 0 
		 * 0 1 2 3 4 
		 * Cluster: 1 
		 * 5 6 7 8 9
		 */
	}
}

/**
 * 
 * @author qyuvks Document class for the vector representation of a document
 */
class Doc {
	// Document ID
	int docID;
	// Term frequency
	double tf;

	public Doc(int docID, double freq) {
		this.docID = docID;
		tf = freq;
	}

	public String toString() {
		return (docID + ":" + tf);
	}
}

/**
 * 
 * @author Saurabh Anant Wani This class creates clusters.
 */
class Cluster {
	ArrayList<Integer> Doc = new ArrayList<Integer>();
	ArrayList<Integer> oldDoc = new ArrayList<Integer>();
	ArrayList<Double> centroid = new ArrayList<Double>();
	ArrayList<ArrayList<Double>> contents = new ArrayList<ArrayList<Double>>();

	// Initialize
	Cluster(int clusterSize) {
		centroid = new ArrayList<Double>(clusterSize);
		for (int j = 0; j < clusterSize; j++)
			centroid.add(0.0);
	}

	public String toString() {
		return oldDoc.toString();
	}

	// To add document to as cluster.
	void addContent(ArrayList<Double> content, int docID) {
		contents.add(content);
		Doc.add(docID);
	}

	// Calculate the cosine similarity between a cluster's centroid and the other
	// cluster.
	Double cosineSimilarity(ArrayList<Double> p1) {
		double s = 0.0;
		double ssp = 0.0;
		double ssc = 0.0;
		for (int i = 0; i < p1.size(); i++) {
			s = s + (p1.get(i) * centroid.get(i));
			ssp = ssp + Math.pow(p1.get(i), 2);
			ssc = ssc + Math.pow(centroid.get(i), 2);
		}
		return s / (Math.sqrt(ssp) * Math.sqrt(ssc));
	}

	// Calculate the centroid based on the documents added to the cluster.
	void compCentroid() {
		for (int i = 0; i < contents.size(); i++) {
			ArrayList<Double> content = contents.get(i);
			for (int j = 0; j < content.size(); j++) {
				centroid.set(j, centroid.get(j) + content.get(j));
			}
		}

		for (int p = 0; p < centroid.size(); p++) {
			centroid.set(p, centroid.get(p) / contents.size());
		}

		oldDoc = new ArrayList<Integer>(Doc);
		Doc = new ArrayList<Integer>();
		contents = new ArrayList<ArrayList<Double>>();
	}
}
