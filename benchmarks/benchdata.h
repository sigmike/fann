#ifndef __BenchData_H__
#define __BenchData_H__

#include <deque>

using std::deque;

class BenchData
{
public:
	BenchData()
		: time(0), epochs(0), mse(0), bitFail(0), 
		  meanBitFail(0), numNeurons(0), numBenchData(0) {}

	BenchData(double time, unsigned int epochs, double mse, unsigned int bitFail, 
		 	  double meanBitFail, unsigned int numNeurons)
		: time(time), epochs(epochs), mse(mse), bitFail(bitFail), 
		  meanBitFail(meanBitFail), numNeurons(numNeurons), numBenchData(1) {}
	
	void printBench(FILE *out)
	{
		if(numBenchData == 1)
			fprintf(out, "%f %.20e %d %d %.20e %d\n", time, mse, epochs, bitFail, meanBitFail, numNeurons);
		else
			fprintf(out, "%f %.20e %f %f %.20e %f %d\n", time/(double)numBenchData, 
				mse/(double)numBenchData, epochs/(double)numBenchData, bitFail/(double)numBenchData, 
				meanBitFail/(double)numBenchData, numNeurons/(double)numBenchData, numBenchData);
	}
	
	void add(BenchData *data)
	{
		time += data->time;	
		epochs += data->epochs;	
		mse += data->mse;	
		bitFail += data->bitFail;	
		meanBitFail += data->meanBitFail;	
		numNeurons += data->numNeurons;	
		numBenchData += data->numBenchData;	
	}
	
	double time;
	unsigned int epochs;
	double mse;
	unsigned int bitFail;
	double meanBitFail;	
	unsigned int numNeurons;
	unsigned int numBenchData;
};

class BenchDataCollector
{
private:
	deque< deque<BenchData *> > collection;
public:
	BenchDataCollector()
	{
		
	}
	
	void addBench(BenchData *data)
	{
		collection[collection.size()-1].push_back(data);
	}
	
	void newCollection()
	{
		collection.push_back(deque<BenchData *>());	
	}
	
	void printAvgCollection(FILE *out, char* name)
	{
		deque<BenchData*> avg;
		
		bool finished = false;
		while(!finished)
		{
			BenchData *avgdata = new BenchData();
			double maxTime = 0;

			//first we get one from each dataset			
			for(deque< deque<BenchData*> >::iterator it = collection.begin(); 
				it != collection.end(); it++)
			{
				if(it->size() == 0)
				{
					//we must have one from each set to make an avg
					finished = true;
					break;
				}
				
				BenchData *data = it->front();
				
				if(data->time > maxTime)
					maxTime = data->time;
				
				avgdata->add(data);
				it->pop_front();
			}
			
			//Then we add all dataset which have a time which is smaller than the maxtime
			for(deque< deque<BenchData*> >::iterator it = collection.begin(); 
				it != collection.end(); it++)
			{
				while(it->size() != 0)
				{
					BenchData *data = it->front();
					
					if(data->time < maxTime)
					{
						avgdata->add(data);
						it->pop_front();
					}
					else
						break; //only take the ones below maxTime						
				}
			}
			
			//last we push the average data on the list
			if(!finished)
				avg.push_back(avgdata);
		}

		BenchData *minData = avg[0];
		for(deque<BenchData*>::iterator it = avg.begin(); it != avg.end(); it++)
		{
			if((*it)->mse < minData->mse)
				minData = (*it);
			
			(*it)->printBench(out);	
		}
		
		FILE *sum = fopen("summary.txt", "a");
		fprintf(sum, "%s ", name);
		minData->printBench(sum);
		fclose(sum);
	}
};

#endif
