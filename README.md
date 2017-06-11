# gmm

![4 component GMM on 4 clusters with kmeans init](http://i.imgur.com/6sMu8dl.png)

This took a lot more effort than expected. See commit history for version that includes a reference 
sklearn GMM (which is also plotted for comparison). Refer to [wiki](https://en.wikipedia.org/wiki/Mixture_model#Expectation_maximization_.28EM.29) or David Mackay's book for formula explanation.

At first I was initializing the means randomly and working directly with the probabilities. It didn't take long 
to realize that random inits were a bad idea, but as everything was so basic I dismissed numerical issues at first and it took
a bit until I realized I'd have to work with log probabilities.

[Here some images.](http://imgur.com/a/iyanF) As the first image shows it did sometimes work, but more often weird stuff happened.

So I started initializing the clusters from subsets of the data and using logprobs. After ironing some kinks out, like not shuffling 
the data (starting means good through cheating) and having code that assumed log(a+b) = log(a) + log(b), things started to come 
together.

Now having more experience I realized that initializing from subsets of the data could lead to it taking a while to converge, as if 
the initial means are close to each other (with similar variances), the responsibilities will be close to even leading to each iteration
moving the means apart very slowly.
At first I did several rounds initializations and then took that which had the means with the highest variance between each other. 
But then I thought I might as well do a hard k-means initialization. This worked very well. 

[Images.](http://imgur.com/a/2hWzF) The first 6 images show 3 different runs, once without and once with kmeans initialization. See run 3 (images 5/6),
now using 3 clusters, for a good an example how kmeans makes convergence faster. Final image shows training on a dataset with 4 clusters. (click on an image for better resolution)
