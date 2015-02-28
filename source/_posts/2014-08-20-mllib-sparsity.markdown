# MLlib：归约数据量

## 以及发发牢骚 :-)

## Table of contents

[TOC]

## 稀疏向量的支持：自然数据中的稀疏属性的挖掘和支持

现在来探索如何在大数据中归约数据量，降低计算复杂度。本文两侧分治之，其一，通过发掘和支持自然数据中的稀疏属性，算作自然资源的挖掘；其二，通过对现有数据的聚集归约，视为人工数据聚合。这两种方法在 MLlib 中均得到了很好的体现。本篇旨在剖析 MLlib 现有的两类数据归约方法，权当为后来的机器学习分布式算法抛砖引玉。

自从 Spark 1.0 以来，MLlib 开始透过`Vector`接口支持稀疏向量。并在下层以 Breeze 承接计算主体。这一改变影响巨大，首先现有的几乎所有算法都遭到了随之而来的改动。从用于矩阵分解的 ALS、SVD，到所有的线性模型，乃至朴素贝叶斯和 K 均值算法都在改动之列。新加入的决策树算法因其起步晚，所以一开始就享受到稀疏向量带来的优势。甚至可以说，正是稀疏矩阵的加入，让 MLlib 由一个 demo 式的玩具，变成了可以工业应用的平台。

### 稀疏性

稀疏性是自然世界的本质属性。在大数据时代的数据几乎是稀疏数据的天下。稀疏性来源于两个方面，一是数据“基”非常大，即数据空间之大；二是数据“点”非常少，即可观测的性征少。N-Gram 是前者，数据空间大小以全体文字数目为幂次，而世界上能出现的N个文字的组合却远少于这个值。点评数据是后者，由于长尾因素，大多数用户能点评的数据非常有限，因此仅填充了“用户——物品”矩阵很少的一部分。

从统计学家，或者机器学习人员的角度来看，现实世界的数据等于稀疏数据（或者低秩数据）加上噪声。而这正是机器学习算法能够成功过的一个重要基础。统计学家和机器学习人员的一个重要工作就是在貌似繁杂的数据中找到那些简单的构造因素。有点类似于牛顿三定律，不论世界多么复杂，物质作用多么繁复，只要是在经典力学的范围内，就要遵循三定律。三定律就是世界的模型，由此构成稀疏的世界。一幅图像是稠密的，其内部充斥着 RGB 三原色的组合，并且大都不为零。一个神经讯号是稠密的，在时间线上连续存在。对于这种数据，我们可以通过小波变换或者傅里叶变换找到它稀疏的一面。

如果把数据点除以数据空间作为数据密度，那么 Netflix 竞赛数据的密度只有 1.17%，rcv1 数据集的数据密度仅 0.15%，近些年来所用到的欺诈检测的数据平均密度仅有 10%，而且这些数据为了提取更多的特征，通常都人为增大了欺诈数据的条目，即是有偏的！现在估算一下你手头数据的数据密度，是否要考虑稀疏性了呢？那么应该怎么做？核心就是要善于发现和利用这种属性。Spark 1.0 加入了稀疏向量，正是往实际应用迈进了临门一脚。

有了稀疏向量只是一个基础，说明我们有能力发掘现实世界的稀疏性了。然而下一步更重要的是，如何在机器学习算法中更好的利用这种性质？要知道，稀疏性无处不在，但又非常脆弱，很多操作就能破坏它。例如向量加法，两个稀疏向量相加的结果会比之前的向量稠密一些，多个向量相加的结果就完全是稠密矩阵了。因此，善于利用利于保持稀疏性的线性代数运算是其中的关键。

<!--more-->

### K 均值的稀疏算法

最典型的成功案例就是 K 均值算法。同样的数据集，在利用稀疏向量之后，不仅省下了大量的存储空间（是数据密度而定），更能让程序加速数倍。K 均值是一个典型的 Expectation-Maximization 算法。即首先在固定聚类中心的前提下求期望（这里是到各中心点的最小距离），然后是固定每个样本类属后最优化中心点（这里就是简单的求均值）。而这个距离计算深深地出卖了 K 均值算法。中心点一般而言都是求均值之后的结果，求均值恰好是一堆稀疏向量求和的过程，因此中心点这个向量一般而言都是稠密的。一个稠密的向量和一个稀疏的样本在求距离的时候很显然是稠密矩阵占优，因为两个向量之间做减法。如此一来，数据本身的稀疏性就被大大的破坏了。为此我们要寻求解决之道。

解决之道就在距离计算公式中，该公式可以拆分为两个 norm 的计算和一个点积的计算。样本本身是稀疏的，因此其 norm 计算也是稀疏的，并且这个值是不变的，可以先期计算好；中心点本身是稠密的，但是其在 expectation 的计算过程中不变，所以只需要计算一次就好；剩下的点积是稀疏向量占优的。这样我们就立马解决了刚才的难题，极大地降低了计算量。

### 线性模型的稀疏算法

下一个比较成功的案例是线性模型。线性模型每次迭代计算的本质是求梯度，而梯度也是样本数据与某种求偏导得来的向量的点积，因此在求每一个样本点的梯度的时候，可以很好的利用稀疏的性质。下面问题来了，梯度在做聚合的时候，如果直接用稀疏向量的加法把这些梯度全部累加起来，其实是非常要命的。首先，根据刚才所言，多个稀疏向量相加最终结果一般是稠密矩阵；其次，根据稀疏向量的定义，要生成很多不必要的临时对象，造成严重的 GC。因此，MLlib 采取的做法是用一个稠密向量作为初始值，将稀疏向量聚合到这个稠密向量上，因此降低了计算的难度。

### 统计值的稀疏性算法

统计值个数、均值、最大最小值、非零元以及方差更是探索稀疏性的便宜之所。首先通过使用流式均值、方差算法，可以再对数据的一遍扫描后得到这六种统计值。其次，由于稀疏性的存在，再次将算法复杂度由 O(nd) 降到 O(nnz)。

### SVD 的稀疏性算法

MLlib 探索稀疏性的最后一例是奇异值分解（SVD）。类似于 ALS，SVD 也是一种常用的推荐算法，并且是由传统的线性代数转换而来。至于 ALS 算法，笔者在 2014 年 8 月期的《程序员》中已作详细的介绍。SVD 在计算上与特征值/特征向量求解是高度相关的，（这里是一些公式）。在使用 Lanczos 算法求解特征值分解的时候，有大量的矩阵与向量乘积运算。因此，在内积计算的时候可以很好的利用稀疏性，而在向量累加的时候利用跟线性模型一样的做法即可。

花开两朵，各表一枝。对于无法直接利用稀疏性来降低计算复杂度的算法来说，人工归约与压缩可能是另外的途径。MLlib 中决策树算法的实现是其中很明显的代表。

## 决策树算法的实现：人为的数据规约降低计算复杂度

### 决策树算法

决策树算法是所有机器学习方法中最容易解释的一种了。依据人工智能的发展史，决策树的祖先可以追溯到上世纪50年代的符号推理模型。John McCarthy等人意图用逻辑与符号表达人类思维过程。而决策树本身可以算作符号推理派和统计学派的“结晶”。

决策树的故事是从一张表开始的。这张表的行是样本，列是特征。每一次决策的过程都会遍历这个表所有的项，从中找到一个最合适的，这次决策就表示为：如果某个样本该特征的值大于当前选择的项，那么属于右边子树中的类，否则属于左边子树中的类。因此决策树其实很简单，站在每个节点上，面对这个节点中有的数据，选择一个“最合适的”表项进行分裂。“最合适的”表项有多种选择，MLlib官方文档就有很详细的介绍，这里不再赘述。

### 数据采样、分割

这里问题就来了。假设数据有千千万，我怎样才能高效的处理这些数据，并从中找到我需要的表项呢？可行的方案之一就是采样，可以使在样本这个级别采样，也可以是在特征这个级别采样，甚至两个方向同时采样。但是这种采样一般都伴随着组合模型，如果单纯训练一棵树，采样比例过低的话会导致效果不好。尤其是，我们无法保证数据本身是随机分布的，因此难以保证随机采样的准确性。

另一个方法是做data parallel，每个数据划分进行自己的决策树训练，最终的结果混合到一起做模型组合。这种想法比较直接，而且可以复用单机的决策树训练算法。但是如果出现了比较极端的情况会导致单个机器上的决策树训练失败。比如每个机器上的数据都不具有可区分性，其各个特征方向方差都很小，而不同的data partition之间方差很大，这样导致单个数据划分内部效果不好。甚至很多情况下都会存在这种问题，即每个数据分片内部数据比较相似，而分片之间数据差异很大。

### 决策树的三大优化算法

于是一个百来行就能写完的程序扩展成将近2000行的“巨无霸”（很多MLlib的程序都面临着这种境况）。**（其实我现在越来越觉得能把简单的程序写的这么复杂不是好事儿，肯定是哪个地方的抽象出问题了。）** MLlib实现的决策树共有三种分布式程序优化方法，一是聚合代替shuffle，二是分桶减少计算，三是逐层训练降低数据扫描次数。下面一一道来。

先看决策树算法的主干（简单起见删了很多不重要的语句）：

```scala
 def train(input: RDD[LabeledPoint]): DecisionTreeModel = {
    val (splits, bins) = DecisionTree.findSplitsBins(retaggedInput, metadata)
    val nodes = new Array[Node](maxNumNodes)

    var level = 0
    var break = false
    while (level <= maxDepth && !break) {
      val splitsStatsForLevel = DecisionTree.findBestSplits(treeInput, parentImpurities,
        metadata, level, nodes, splits, bins, maxLevelForSingleGroup, timer)
      for ((nodeSplitStats, index) <- splitsStatsForLevel.view.zipWithIndex) {
        extractNodeInfo(nodeSplitStats, level, index, nodes)
      }
    }

    val topNode = nodes(0)
    topNode.build(nodes)

    new DecisionTreeModel(topNode, strategy.algo)
  }
```

通过这个还算清晰的框架，可以看出决策树训练共有三步，一是`findSplitsBins`，其作用是将原始数据按照特征的维度分桶。二是while循环中的`findBestSplits`，目的是在决策树每层节点的训练中为每个节点找到最佳的分裂点。最后一步是构造决策树。

现在寻根溯源，首先看看数据按特征分桶是怎么做的：

```scala
 protected[tree] def findSplitsBins(
      input: RDD[LabeledPoint],
      metadata: DecisionTreeMetadata): (Array[Array[Split]], Array[Array[Bin]]) = {
      
    val requiredSamples = numBins*numBins
    val fraction = if (requiredSamples < count) requiredSamples.toDouble / count else 1.0
    
    val sampledInput =
      input.sample(withReplacement = false, fraction, new XORShiftRandom().nextInt()).collect()
    val numSamples = sampledInput.length

    val stride: Double = numSamples.toDouble / numBins

    metadata.quantileStrategy match {
      case Sort =>
        val splits = Array.ofDim[Split](numFeatures, numBins - 1)
        val bins = Array.ofDim[Bin](numFeatures, numBins)

        // Find all splits.

        // Iterate over all features.
        var featureIndex = 0
        while (featureIndex < numFeatures) {
          // Check whether the feature is continuous.
          val isFeatureContinuous = metadata.isContinuous(featureIndex)
          if (isFeatureContinuous) {
            val featureSamples = sampledInput.map(lp => lp.features(featureIndex)).sorted
            val stride: Double = numSamples.toDouble / numBins
            logDebug("stride = " + stride)
            for (index <- 0 until numBins - 1) {
              val sampleIndex = index * stride.toInt
              // Set threshold halfway in between 2 samples.
              val threshold = (featureSamples(sampleIndex) + featureSamples(sampleIndex + 1)) / 2.0
              val split = new Split(featureIndex, threshold, Continuous, List())
              splits(featureIndex)(index) = split
            }
          } else { ??? }
        }

        // Find all bins.
        featureIndex = 0
        while (featureIndex < numFeatures) {
          val isFeatureContinuous = metadata.isContinuous(featureIndex)
          if (isFeatureContinuous) { // Bins for categorical variables are already assigned.
            bins(featureIndex)(0) = new Bin(new DummyLowSplit(featureIndex, Continuous),
              splits(featureIndex)(0), Continuous, Double.MinValue)
            for (index <- 1 until numBins - 1) {
              val bin = new Bin(splits(featureIndex)(index-1), splits(featureIndex)(index),
                Continuous, Double.MinValue)
              bins(featureIndex)(index) = bin
            }
            bins(featureIndex)(numBins-1) = new Bin(splits(featureIndex)(numBins-2),
              new DummyHighSplit(featureIndex, Continuous), Continuous, Double.MinValue)
          }
          featureIndex += 1
        }
        (splits, bins)
        
      case _ => ???
    }
  }
```

Bin化（又叫数据规约、分桶）这个地方比较单纯，很少调用其他的函数。这里要先界定两个名词，split 和 bin。前者表示分裂的点，后者表示分裂的点之间的线段（样本集）。由于 bin 的个数通常都是有限制的，因此即便`requiredSamples`数目是 bin 个数的平方，也是可以将这些采样到的数据收回到 Driver 端的，即`sampledInput`。Bin 化采样的这个地方有点类似于 TeraSort。之后的第一个任务是根据采样的结果找到每个特征的可行的分裂点（split），这里就看出采样的作用来了，它实际上是一种数据规约，限定了分裂点只能由采样结果计算得来。代码中的`while`循环针对于每一个 feature，对于每个 feature，我们根据前面算出来的`stride`（步长）信息计算出一个个的分裂点。

代码的下半部分就是根绝得到的 split，找到其对应的所有的 bin。把 split 理解成点，则 bin 就是两点之间的线段，因此，只要给定两个端点就能确定一个 bin。遇到开头和结尾的地方，会用哑元`DummyLowSplit`和`DummyHighSplit`代替。

划分了 split 和 bin 最大的好处就是分裂点由数据表中所有的表项减少到所有的 split，而在计算一些统计量的时候 bin 可以减少计算量，每个 bin 内部直接归约。

下一步就是在 split 之中寻找最佳的分裂点。这里是一个循环，每次循环都会完成一棵树的一个完整的层次的节点的计算。当然，随着树的深度不断加深，节点的数目成倍增长，因此这里用到了一点小小的技巧：设定一个最大可同时处理的节点数目，如果那一层的节点数超过了这个范围，那么就层内分组进行训练，因此下面我们不再分析`findBestSplits`的代码，转而直接分析它所调用的`findBestSplitsPerGroup`。

```scala
  private def findBestSplitsPerGroup(
      input: RDD[TreePoint],
      parentImpurities: Array[Double],
      metadata: DecisionTreeMetadata,
      level: Int,
      nodes: Array[Node],
      splits: Array[Array[Split]],
      bins: Array[Array[Bin]],
      timer: TimeTracker,
      numGroups: Int = 1,
      groupIndex: Int = 0): Array[(Split, InformationGainStats)] = {

    val bestSplits = new Array[(Split, InformationGainStats)](numNodes)
   
    var node = 0
    while (node < numNodes) {
      val nodeImpurityIndex = (1 << level) - 1 + node + groupShift
      val binsForNode: Array[Double] = getBinDataForNode(node)
      val parentNodeImpurity = parentImpurities(nodeImpurityIndex)
      bestSplits(node) = binsToBestSplit(binsForNode, parentNodeImpurity)
      node += 1
    }
    bestSplits
  }
```

面对一个 800 来行的程序，压力山大。首先把里面的函数定义、注释、log 全部删掉，先看主体，稍后再慢慢关注内部函数细节。首先，`bestSplits`是个数组，意味着决策树的当前层所有的节点都会计算一个最佳分裂点。下面的 while 循环就是对每个节点找到最佳分裂点。其中`binsForNode`的意思是找到当前节点中所有的 bin。如果把决策树的每个节点看做一个过滤，那么越往叶子节点走，每个节点所有的数据就越少，因为被过滤掉的越多。最后由`binsToBestSplit`这个函数找到当前节点中所有 bin 中选出的最佳分裂点。

看清了主干，下面按图索骥，先看`getBinDataForNode`是怎么回事：

```scala
    val binMappedRDD = input.map(x => findBinsForLevel(x))
    
    val binAggregates = {
      binMappedRDD.aggregate(Array.fill[Double](binAggregateLength)(0))(binSeqOp,binCombOp)
    }
    
    def getBinDataForNode(node: Int): Array[Double] = {
      if (metadata.isClassification) { 
        ???
      } else {
        // Regression
        val shift = 3 * node * numBins * numFeatures
        val binsForNode = binAggregates.slice(shift, shift + 3 * numBins * numFeatures)
        binsForNode
      }
    }

```
如果只看`getBinDataForNode`的逻辑的话，其实很简单，每个节点的`binsForNode`是由`binAggregates`这个变量中切了一块出来得到的。切的地方是`3 * node * numBins * numFeatures`，一共切掉`3 * numBins * numFeatures`这么大。所以关键在`binAggregates`是怎么生成的。

首先要说明`binAggregates`这个变量内部都有些什么吧。这个变量其实是个`Array[Double]`，其中约定好了结构，即对于每个节点中的每个 feature 的每个 bin 计算三个统计量，分别为 count，sum，以及sum^2。统计这三个量的意思很明显，就是均值和方差。这也解释了上面那个貌似奇怪的切分边界。具体原始数据是怎么分割到每个节点的呢？答案在`findBinsForLevel`这个函数中。

```scala
    def findBinsForLevel(labeledPoint: LabeledPoint): Array[Double] = {
      // Calculate bin index and label per feature per node.
      val arr = new Array[Double](1 + (numFeatures * numNodes))
      arr(0) = labeledPoint.label
      var nodeIndex = 0
      while (nodeIndex < numNodes) {
        val parentFilters = findParentFilters(nodeIndex)
        // Find out whether the sample qualifies for the particular node.
        val sampleValid = isSampleValid(parentFilters, labeledPoint)
        val shift = 1 + numFeatures * nodeIndex
        if (!sampleValid) {
          // Mark one bin as -1 is sufficient.
          arr(shift) = InvalidBinIndex
        } else {
          var featureIndex = 0
          while (featureIndex < numFeatures) {
            val isFeatureContinuous = strategy.categoricalFeaturesInfo.get(featureIndex).isEmpty
            arr(shift + featureIndex) = findBin(featureIndex, labeledPoint,isFeatureContinuous)
            featureIndex += 1
          }
        }
        nodeIndex += 1
      }
      arr
    }
```

这个函数的输入是原始数据中的每一个`LabeledPoint`，输出结果是一个数组，表示当前样本所有 feature 的所有 bin 划分。最终整体的`binMappedRDD`作用是确定每个样本的每个 feature 应该落到哪个 bin 里面。

剩下的事情要看`binSeqOp`和`binComOp`这两个函数了。这两个函数式`aggregate`算子的参数。前者表明在一个数据分片内如何统计数据，后这说明两个分片的结果如何做合并。下面我们剥离出一些函数来看：

```scala
    def regressionBinSeqOp(arr: Array[Double], agg: Array[Double]) {
      // Iterate over all nodes.
      var nodeIndex = 0
      while (nodeIndex < numNodes) {
        // Check whether the instance was valid for this nodeIndex.
        val validSignalIndex = 1 + numFeatures * nodeIndex
        val isSampleValidForNode = arr(validSignalIndex) != InvalidBinIndex
        if (isSampleValidForNode) {
          // actual class label
          val label = arr(0)
          // Iterate over all features.
          var featureIndex = 0
          while (featureIndex < numFeatures) {
            // Find the bin index for this feature.
            val arrShift = 1 + numFeatures * nodeIndex
            val arrIndex = arrShift + featureIndex
            // Update count, sum, and sum^2 for one bin.
            val aggShift = 3 * numBins * numFeatures * nodeIndex
            val aggIndex = aggShift + 3 * featureIndex * numBins + arr(arrIndex).toInt * 3
            agg(aggIndex) = agg(aggIndex) + 1
            agg(aggIndex + 1) = agg(aggIndex + 1) + label
            agg(aggIndex + 2) = agg(aggIndex + 2) + label*label
            featureIndex += 1
          }
        }
        nodeIndex += 1
      }
    }
```

```scala
    def binCombOp(agg1: Array[Double], agg2: Array[Double]): Array[Double] = {
      var index = 0
      val combinedAggregate = new Array[Double](binAggregateLength)
      while (index < binAggregateLength) {
        combinedAggregate(index) = agg1(index) + agg2(index)
        index += 1
      }
      combinedAggregate
    }
```

看了这两个函数，是不是感觉一目了然？前者把三个统计量聚合在`agg`这个数组中，后者来合并多个`agg`数组。

现在决策树当前层上所有节点都得到自己节点中数据的统计量了，下一步就是为每个节点寻找最佳分裂点了：

```scala
    def binsToBestSplit(
        binData: Array[Double],
        nodeImpurity: Double): (Split, InformationGainStats) = {

      logDebug("node impurity = " + nodeImpurity)

      // Extract left right node aggregates.
      val (leftNodeAgg, rightNodeAgg) = extractLeftRightNodeAggregates(binData)

      // Calculate gains for all splits.
      val gains = calculateGainsForAllNodeSplits(leftNodeAgg, rightNodeAgg, nodeImpurity)

      val (bestFeatureIndex,bestSplitIndex, gainStats) = {
        // Initialize with infeasible values.
        var bestFeatureIndex = Int.MinValue
        var bestSplitIndex = Int.MinValue
        var bestGainStats = new InformationGainStats(Double.MinValue, -1.0, -1.0, -1.0, -1.0)
        // Iterate over features.
        var featureIndex = 0
        while (featureIndex < numFeatures) {
          // Iterate over all splits.
          var splitIndex = 0
          while (splitIndex < numBins - 1) {
            val gainStats = gains(featureIndex)(splitIndex)
            if (gainStats.gain > bestGainStats.gain) {
              bestGainStats = gainStats
              bestFeatureIndex = featureIndex
              bestSplitIndex = splitIndex
            }
            splitIndex += 1
          }
          featureIndex += 1
        }
        (bestFeatureIndex, bestSplitIndex, bestGainStats)
      }

      logDebug("best split bin = " + bins(bestFeatureIndex)(bestSplitIndex))
      logDebug("best split bin = " + splits(bestFeatureIndex)(bestSplitIndex))

      (splits(bestFeatureIndex)(bestSplitIndex), gainStats)
    }
```

至此看似复杂的决策树已经抽丝剥茧完毕。回顾一下，三个程序优化方法分别是

* 以 aggregate 代替 shuffle 来计算统计量；
* 以 bin 来代替原始数据归约数据量；
* 层次化的模型训练代替逐节点的模型训练。

其中最重要的是第二个优化，也就是本节重点想说的：**人为的数据规约降低计算复杂度**。

## 结语

最近有点懒，现在把拖了很久的这篇文章整理一下，聊表慰藉。本文原创的思想来源于 Spark Summit 2014 Xiangrui 和 Amde 两个讲座的 slide，都很有启发和借鉴的意义，因此整理一下放在这里。做了一年的 Spark，尤其是 MLlib，现在回头来看，为什么在 Spark 平台上写一个优秀的机器学习算法这么难？（这样说有点不公平，因为在 Hadoop 上实现会更烦琐。）或许真的是现有 Spark 的抽象不适合机器学习，但是迄今为止开源社区没有更好的分布式框架的抽象可以用了。也许各大互联网公司有自己的秘密武器，不过这就不得而知了。MLlib 的本意（从名字上来看）也许是想解放 machine learner，但是现在越来越走在解放 end user的道路上了。但我始终觉得这不是良策，因为 machine leaner 开发新的机器学习程序还是很难，这样导致整个系统不具有可扩展性。也许对于机器学习没有 all in one 这种让我们省力的抽象或者平台，也许 machine learner 就该在平台和算法的世界里继续努力的奋斗。

无论如何，好的思想值得借鉴和反思，也许本文值得一看！

> Written with [StackEdit](https://stackedit.io/).
