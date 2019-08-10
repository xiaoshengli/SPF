//Copyright (c) 2019 Xiaosheng Li (xli22@gmu.edu)
//Reference: Linear Time Complexity Time Series Clustering with Symbolic Pattern Forest, IJCAI 2019
/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include<vector>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "metis.h"

#define INF 1e20
#define MAX_PER_LINE 1000000

#define ROUNDNUM(x) ((int)(x + 0.5))

using namespace std;

void loadDataset(char* trainFileName, char* testFileName, vector<vector<double> > &train, vector<double> &trainLabel) {
	FILE *trainp;
	char buf[MAX_PER_LINE];
	char* tmp;
	double label;

	trainp = fopen(trainFileName, "r");
	if (trainp == NULL) {
		cout << "Error!! Input file " << trainFileName << " not found." << endl;
		exit(0);
	}

	while (fgets(buf, MAX_PER_LINE, trainp)) {
		vector<double> ts;
		tmp = strtok(buf, ", \t\r\n");
		label = atof(tmp);
		trainLabel.push_back(label);
		tmp = strtok(NULL, ", \t\r\n");
		while (tmp != NULL) {
			ts.push_back(atof(tmp));
			tmp = strtok(NULL, ", \t\r\n");
		}
		train.push_back(ts);
	};
	fclose(trainp);

	trainp = fopen(testFileName, "r");
	if (trainp == NULL) {
		cout << "Error!! Input file " << testFileName << " not found." << endl;
		exit(0);
	}

	while (fgets(buf, MAX_PER_LINE, trainp)) {
		vector<double> ts;
		tmp = strtok(buf, ", \t\r\n");
		label = atof(tmp);
		trainLabel.push_back(label);
		tmp = strtok(NULL, ", \t\r\n");
		while (tmp != NULL) {
			ts.push_back(atof(tmp));
			tmp = strtok(NULL, ", \t\r\n");
		}
		train.push_back(ts);
	};
	fclose(trainp);
}

int* adjustLabelSet(const double* trainLabel, double* trainLabelIndex, vector<double> &tlabel, int m) {
	unordered_set<double> labelset;
	int i;

	for (i = 0; i < m; ++i) {
		labelset.insert(trainLabel[i]);
	}

	tlabel.assign(labelset.begin(), labelset.end());

	sort(tlabel.begin(), tlabel.end());

	int* classCount = (int *)malloc(sizeof(int) * (tlabel.size()));
	fill(classCount, classCount + tlabel.size(), 0);

	for (i = 0; i < m; ++i) {
		const int position = find(tlabel.begin(), tlabel.end(), trainLabel[i]) - tlabel.begin();
		trainLabelIndex[i] = position;
		classCount[position]++;
	}

	return classCount;
}

void cumsum(double* train, double* cumTrain, double* cumTrain2, int m, int n) {
	int i, j;
	double* pt = train;
	double* pcx = cumTrain;
	double* pcx2 = cumTrain2;
	for (i = 0; i < m; ++i) {
		*(pcx++) = 0;
		*(pcx2++) = 0;
	}
	double* sum;
	sum = (double *)malloc(sizeof(double) * (m));
	double* psum = sum;
	for (i = 0; i < m; ++i) {
		*(psum++) = 0;
	}
	double* sum2;
	sum2 = (double *)malloc(sizeof(double) * (m));
	double* psum2 = sum2;
	for (i = 0; i < m; ++i) {
		*(psum2++) = 0;
	}

	for (j = 0; j < n; ++j) {
		psum = sum;
		psum2 = sum2;
		for (i = 0; i < m; ++i) {
			*psum += *pt;
			*psum2 += (*pt)*(*pt);
			*(pcx++) = *psum;
			*(pcx2++) = *psum2;
			psum++;
			psum2++;
			pt++;
		}
	}

	free(sum);
	free(sum2);
}

void indicating(const double* train, int wd, int wl, const double* cumTrain, const double* cumTrain2, int m, int n, bool* indicating_array) {
	const double* pmat = train;
	const double* pcx = cumTrain;
	const double* pcx2 = cumTrain2;
	int i, j, k, pword, u, l;

	const int symbolicsize = 1 << (2 * wd);
	const double ns = (1.0 * wl) / wd;

	bool* pr = indicating_array;
	for (i = 0; i < m*symbolicsize; ++i) {
		*(pr++) = false;
	}
	pr = indicating_array;

	for (i = 0; i<m; ++i) {
		pword = -1;
		for (j = 0; j<n - wl + 1; ++j) {
			const double sumx = *(pcx + i + (j + wl)*m) - *(pcx + i + j*m);
			const double sumx2 = *(pcx2 + i + (j + wl)*m) - *(pcx2 + i + j*m);
			const double meanx = sumx / wl;
			const double sigmax = sqrt(sumx2 / wl - meanx*meanx);
			int wordp = 0;
			for (k = 0; k<wd; ++k) {
				u = ROUNDNUM(ns*(k + 1));
				l = ROUNDNUM(ns*k);
				const double sumsub = *(pcx + i + (j + u)*m) - *(pcx + i + (j + l)*m);
				const double avgsub = sumsub / (u - l);
				const double paa = (avgsub - meanx) / sigmax;
				int val;
				if (paa < 0)
					if (paa < -0.67) val = 0;
					else val = 1;
				else
					if (paa < 0.67) val = 2;
					else val = 3;
					const int ws = (1 << (2 * k))*val;
					wordp += ws;
			}
			if (pword != wordp) {
				(*(pr + i + wordp*m)) = true;
				pword = wordp;
			}
		}
	}
}

void count_pattern(const bool* indicating_array, int m, int symbolicsize, int* pattern_count) {
	int i, count = 0, tcount = 0;
	const bool* pmat = indicating_array;
	int* pc = pattern_count;
	for (i = 0; i < m*symbolicsize; ++i) {
		if (*(pmat++)) tcount++;
		count++;
		if (count == m) {
			count = 0;
			*pc = tcount;
			tcount = 0;
			pc++;
		}
	}
}

int* get_index(const int* pattern_count, int symbolicsize, int p_low, int p_high, int num_p) {
	int* p_index = (int *)malloc(sizeof(int) * num_p);
	const int* pc = pattern_count;
	int* pr = p_index;
	for (int i = 0; i < symbolicsize; ++i) {
		if (*pc > p_low) {
			if (*pc < p_high) *(pr++) = i;
		}
		pc++;
	}
	return p_index;
}

int find_candidates(const int* pattern_count, int symbolicsize, int p_low, int p_high) {
	int count = 0;
	const int* pc = pattern_count;
	for (int i = 0; i < symbolicsize; ++i) {
		if (*pc > p_low) {
			if (*pc < p_high) count++;
		}
		pc++;
	}
	return count;
}

void vecs2arr(const vector<vector<double> > &trainV, double* train) {
	int i, j, m, n;
	double *pt = train;
	m = trainV.size();
	n = trainV[0].size();
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			*(pt++) = trainV[i][j];
		}
	}
}

void vec2arr(const vector<double> &trainLabelV, double* trainLabel) {
	int i, m;
	double *pt = trainLabel;
	m = trainLabelV.size();
	for (i = 0; i < m; i++) {
		*(pt++) = trainLabelV[i];
	}
}

void SPT(const bool* indicating_array, int* p_index, int m, int num_p, int* subclusters, int k, int c) {
	int i, j, tnum, fnum;
	srand((int)time(NULL));
	random_shuffle(p_index, p_index + num_p);
	const int* pi = p_index;
	int* pst = subclusters + m*k;
	int* ps;
	for (i = 0; i < c; ++i) {
		if (i == c - 1) {
			ps = pst;
			for (j = 0; j < m; ++j) {
				if (*ps == -1) *ps = i;
				ps++;
			}
			break;
		}
		const bool* pt = indicating_array + (*pi) * m;
		tnum = 0;
		fnum = 0;
		for (j = 0; j < m; ++j) {
			if (*pt) tnum++;
			else fnum++;
			pt++;
		}
		pt = indicating_array + (*pi) * m;
		ps = pst;
		if (tnum > fnum) {
			for (j = 0; j < m; ++j) {
				if (*ps == -1)
					if (!*pt) *ps = i;
				pt++;
				ps++;
			}
		}
		else {
			for (j = 0; j < m; ++j) {
				if (*ps == -1)
					if (*pt) *ps = i;
				pt++;
				ps++;
			}
		}
		pi++;
	}
}

void ensemble2graph(int* subclusters, const int m, const int en_num, idx_t* xadj, idx_t* adjncy, const int c) {
	int i, j, k, end = 0;
	int* psub = subclusters;
	idx_t* pe = xadj;
	idx_t* pad = adjncy;
	*pe = 0;
	pe++;
	for (i = 0; i < m*en_num; ++i) {
		*(psub + i) = *(psub + i) + m + c*(i / m);
	}
	for (i = 0; i < m; ++i) {
		for (j = 0; j < en_num; ++j) {
			*pad = *(psub + i + j*m);
			pad++;
			end++;
		}
		*pe = end;
		pe++;
	}
	for (i = 0; i < en_num; ++i) {
		int* ps = psub + i*m;
		for (j = 0; j < c; ++j) {
			const int v = m + i*c + j;
			for (k = 0; k < m; ++k) {
				if (*(ps + k) == v){
					*pad = k;
					pad++;
					end++;
				}
			}
			*pe = end;
			pe++;
		}
	}
}

void collectPart(int* clusters, const idx_t* part, int pos, int m) {
	int* pcls = clusters + pos*m;
	const idx_t* pp = part;
	for (int i = 0; i < m; ++i) {
		*pcls = *pp;
		pcls++;
		pp++;
	}
}

void collectClusters(const int* clusters, int* effectiveClusters, vector<int>& effective, int m) {
	int i, j;
	int* pecl = effectiveClusters;
	for (i = 0; i < effective.size(); ++i) {
		const int pos = effective[i];
		const int* pcl = clusters + pos*m;
		for (j = 0; j < m; ++j) {
			*pecl = *pcl;
			pecl++;
			pcl++;
		}
	}
}

double randIndex(const double* trainLabel, const idx_t* part2, int m) {
	int i, j, tp = 0, fp = 0, tn = 0, fn = 0;
	for (i = 0; i < m - 1; ++i) {
		for (j = i + 1; j < m; ++j) {
			if (*(trainLabel + i) == *(trainLabel + j)) {
				if (*(part2 + i) == *(part2 + j)) tp++;
				else fn++;
			}
			else {
				if (*(part2 + i) == *(part2 + j)) fp++;
				else tn++;
			}
		}
	}
	return 1.0 * (tp + tn) / (tp + tn + fp + fn);
}

int main(int argc, char *argv[]) {

	string dataset = string(argv[1]);
	const int en_num = atoi(argv[2]);

	//string path = "./UCR_TS_Archive_2015/";
	//string path2 = "./UCR_TS_Archive_2015/";
	string path = "./";
	string path2 = "./";
	char trainFileName[200];
	char testFileName[200];
	strcpy(trainFileName, path.append(dataset).append("/").append(dataset).append("_TRAIN").c_str());
	strcpy(testFileName, path2.append(dataset).append("/").append(dataset).append("_TEST").c_str());

	cout << "dataset: " << dataset << ", ensemble size: " << en_num << endl;

	int m, n, i, j, wl, wd, k, c;
	double tStart, tEnd;
	vector<vector<double> > trainV;
	vector<double> trainLabelV;

	//load data, here train varaibles contain both training and testing data from the UCR datasets
	loadDataset(trainFileName, testFileName, trainV, trainLabelV);

	n = trainV[0].size();
	m = trainV.size();

	double* train = (double *)malloc(sizeof(double) * (m*n));
	double* trainLabel = (double *)malloc(sizeof(double) * m);

	vecs2arr(trainV, train);
	vec2arr(trainLabelV, trainLabel);

	tStart = clock();

	vector<double> tlabel;
	double* trainLabelIndex = (double *)malloc(sizeof(double) * m);
	int* classCount = adjustLabelSet(trainLabel, trainLabelIndex, tlabel, m);
	c = tlabel.size();

	double* cumTrain = (double *)malloc(sizeof(double) * (m*(n + 1)));
	double* cumTrain2 = (double *)malloc(sizeof(double) * (m*(n + 1)));

	//calculate the cummulative sums
	cumsum(train, cumTrain, cumTrain2, m, n);

	int wdArray[] = { 3, 4, 5, 6, 7 };
	vector<int> wdList(wdArray, wdArray + 5);

	vector<int> wlList;
	wlList.reserve(40);
	for (i = 0; i < 40; ++i) {
		wl = ROUNDNUM((i + 1)*0.025*n);
		if (wl >= 10) wlList.push_back(wl);
	}
	wlList.erase(unique(wlList.begin(), wlList.end()), wlList.end());

	int wdNum = wdList.size();
	int wlNum = wlList.size();

	int totalNum = wdNum * wlNum;
	int maxsymbolicsize = (1 << (2 * wdList[wdNum - 1]));
	bool* indicating_array = (bool *)malloc(sizeof(bool) * (m*maxsymbolicsize));
	int* pattern_count = (int *)malloc(sizeof(int) * maxsymbolicsize);
	const double p_rate = 0.25;
	const int p_low = ROUNDNUM(p_rate * m / c);
	const int p_high = m - p_low;
	int* clusters = (int *)malloc(sizeof(int) * (m*totalNum));
	int* subclusters = (int *)malloc(sizeof(int) * (m*en_num));

	idx_t nvtxs = m + c*en_num;
	idx_t ncon = 1;
	idx_t *xadj = (idx_t *)malloc(sizeof(idx_t) * (m + c*en_num + 1));
	idx_t *adjncy = (idx_t *)malloc(sizeof(idx_t) * (2 * m*en_num));
	idx_t nparts = c;
	idx_t objval;
	idx_t* part = (idx_t *)malloc(sizeof(idx_t) * nvtxs);
	vector<int> effective;

	//grid seach on the parameter combinations
	for (i = 0; i < wdNum; ++i) {
		wd = wdList[i];
		const int symbolicsize = (1 << (2 * wd));
		for (j = 0; j < wlNum; ++j) {
			wl = wlList[j];
			const int pos = i * wlNum + j;

			//generate the boolean indicating arrays
			indicating(train, wd, wl, cumTrain, cumTrain2, m, n, indicating_array);

			//count symbolic patterns
			count_pattern(indicating_array, m, symbolicsize, pattern_count);
			int num_p = find_candidates(pattern_count, symbolicsize, p_low, p_high);

			//select symbolic pattern candidates
			int* p_index = get_index(pattern_count, symbolicsize, p_low, p_high, num_p);
			fill(subclusters, subclusters + (m*en_num), -1);
			if (num_p < c) continue;

			//SPT procedures
			for (k = 0; k < en_num; ++k) {
				SPT(indicating_array, p_index, m, num_p, subclusters, k, c);
			}

			//ensemble procedure
			ensemble2graph(subclusters, m, en_num, xadj, adjncy, c);
			int flag = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy,
				NULL, NULL, NULL, &nparts, NULL,
				NULL, NULL, &objval, part);

			if (flag != METIS_OK) {
				cout << "some errors happen!" << endl;
				continue;
			}

			collectPart(clusters, part, pos, m);
			effective.push_back(pos);
			free(p_index);
		}
	}

	free(indicating_array);
	free(pattern_count);
	free(subclusters);
	free(xadj);
	free(adjncy);
	free(part);

	int en_num2 = effective.size();
	int* effectiveClusters = (int *)malloc(sizeof(int) * (m*en_num2));
	collectClusters(clusters, effectiveClusters, effective, m);

	idx_t nvtxs2 = m + c*en_num2;
	idx_t ncon2 = 1;
	idx_t *xadj2 = (idx_t *)malloc(sizeof(idx_t) * (m + c*en_num2 + 1));
	idx_t *adjncy2 = (idx_t *)malloc(sizeof(idx_t) * (2 * m*en_num2));
	idx_t nparts2 = c;
	idx_t objval2;
	idx_t* part2 = (idx_t *)malloc(sizeof(idx_t) * nvtxs2);

	//final ensemble
	ensemble2graph(effectiveClusters, m, en_num2, xadj2, adjncy2, c);

	int flag = METIS_PartGraphKway(&nvtxs2, &ncon2, xadj2, adjncy2,
		NULL, NULL, NULL, &nparts2, NULL,
		NULL, NULL, &objval2, part2);

	if (flag != METIS_OK) {
		cout << "error!" << endl;
	}

	tEnd = clock();

	//calculate the rand index
	double ranIndex = randIndex(trainLabel, part2, m);

	cout << "rand index: " << ranIndex << endl;
	cout << "The running time is: " << fixed << (tEnd - tStart) / CLOCKS_PER_SEC << "seconds" << endl;

	free(xadj2);
	free(adjncy2);
	free(part2);
	free(train);
	free(trainLabel);
	free(trainLabelIndex);
	free(cumTrain);
	free(cumTrain2);
	free(clusters);
	free(effectiveClusters);
	free(classCount);

	return 0;
}
