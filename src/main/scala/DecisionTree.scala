/**
 * Find the structure of an optimal decision tree based on an input set of decision rules with boolean attributes.
 * Uses the principle of minimization of maximal remaining entropy. Works in a depth-first fashion.
 *
 * @note This was transformed from a REPL script I wrote while learning about the algorithm,
 *       it does need further refactoring and likely some restructuring of the recursion :)
 *       In the current state the terminal nodes and their outcomes need to be interpreted manually.
 */
object DecisionTree {
  private case class Attribute(id: Int,
                               possibleValues: Vector[Int])

  private case class DecisionRule(conclusion: Int,
                                  probability: Double,
                                  elementaryDecisions: collection.immutable.Map[Attribute, Int])

  def main(args: Array[String]): Unit = {
    // INPUT:
    val a1 = Attribute(1, Vector(0, 1))
    val a2 = Attribute(2, Vector(0, 1))
    val a3 = Attribute(3, Vector(0, 1))

    val d1 = DecisionRule(1, 0.15, Map(a1 -> 0, a2 -> 0, a3 -> 0))
    val d2 = DecisionRule(1, 0.15, Map(a1 -> 1, a2 -> 0, a3 -> 0))
    val d3 = DecisionRule(2, 0.2, Map(a1 -> 0, a2 -> 0, a3 -> 1))
    val d4 = DecisionRule(2, 0.2, Map(a1 -> 0, a2 -> 1, a3 -> 0))
    val d5 = DecisionRule(2, 0.2, Map(a1 -> 0, a2 -> 1, a3 -> 1))
    val d6 = DecisionRule(3, 0.05, Map(a1 -> 1, a2 -> 0, a3 -> 1))
    val d7 = DecisionRule(3, 0.05, Map(a1 -> 1, a2 -> 1, a3 -> 1))

    implicit val decisionRules: Vector[DecisionRule] = Vector(d1, d2, d3, d4, d5, d6, d7)

    findOptimalTree(Vector(a1, a2, a3), List())
  }

  def findOptimalTree(remainingAttributes: Vector[Attribute], path: List[(Attribute, Int)])(implicit decisionRules: Vector[DecisionRule]): Unit = {
    println("\n____________________________________")
    println("Pool of remaining attributes: " + remainingAttributes.map(_.id).toList)
    print("Current path:  ")
    println(path.map(x => s"Attribute ${x._1.id} ---${x._2}--->").mkString(" "))

    ///// STEP 1: Find out min of maximum remaining entropy for each attribute in the pool of remaining attributes, add the node to path
    val matrix: Seq[(Int, Int, Int, Double)] = for (a <- remainingAttributes;
                                                    pv <- a.possibleValues;
                                                    dr <- decisionRules)
      yield (a.id,
        pv,
        dr.conclusion,
        if (dr.elementaryDecisions(a) == pv && path.toSet.subsetOf(dr.elementaryDecisions.toSet)) dr.probability else 0)

    // Map: (attributeId, attributeValue, conclusion) -> probability
    val probabilities: Map[(Int, Int, Int), Double] = matrix.groupBy {
      case (attributeId, value, conclusion, probability) => (attributeId, value, conclusion)
    }.view
      .mapValues(x => x.foldLeft(0.0)((a, b) => a + b._4))
      .toMap

    // Map: (attributeId, attributeValue) -> probability
    val probabilities2: Map[(Int, Int), Double] = matrix.groupBy {
      case (attributeId, value, conclusion, probability) => (attributeId, value)
    }.view
      .mapValues(x => x.foldLeft(0.0)((a, b) => a + b._4))
      .toMap

    // Map: (attributeId, attributeValue, conclusion) -> conditional probability (conclusion | attributeValue & attributeId)
    val conditionalProbabilities: Map[(Int, Int, Int), Double] = probabilities.map {
      case ((attributeId, value, conclusion), probability) =>
        (attributeId, value, conclusion) -> (probability / probabilities2.get(attributeId, value).get)
    }
    println("\nConditional probabilities: ")
    conditionalProbabilities.toVector.sortBy(x => x).foreach(x => println(s"Probability of Conclusion ${x._1._3} given Attribute ${x._1._1} and its Value ${x._1._2} = ${x._2}"))

    val remainingEntropy: Map[Int, Map[(Int, Int), Double]] = conditionalProbabilities.map {
      case ((attributeId, value, conclusion) -> (cp)) =>
        (attributeId, value, conclusion) -> (if (cp == 0.0) 0.0 else -cp * Math.log(cp) / Math.log(2))
    }.groupBy { case ((attributeId, value, conclusion), probability) => (attributeId, value)
    }.view
      .mapValues(x => x.values.sum)
      .toMap
      .groupBy { case ((attributeId, value), entropy) => attributeId }
      .toMap
    println("\nRemaining entropy: ")
    remainingEntropy.toVector.sortBy(x => x._1).foreach(x => x._2.toVector.foreach(y =>
      println(s"Remaining entropy in branch with value ${y._1._2} if we pick Attribute ${x._1} = ${y._2}")))

    // Max remaining entropy (between the attribute's values) if attribute is chosen for splitting
    val maxRemainingEntropy = remainingEntropy
      .view
      .mapValues(x => x.values.max)
      .toMap
    println("\nMax remaining entropy if attribute is chosen for splitting: ")
    maxRemainingEntropy.toVector.sortBy(x => x).foreach(x => println(s"Attribute ${x._1}: ${x._2}"))

    val optimalNode: (Int, Double) = maxRemainingEntropy.minBy(x => x._2)
    val optimalAttribute: Attribute = remainingAttributes.filter(a => a.id == optimalNode._1).apply(0)
    ///// STEP 2: If min(max remaining entropy) == 0, then this is a terminal/leaf node, else run for child values
    // TODO NaN (result of division by 0 when computing the conditional probability) is just a
    //  shortcut instead of a proper implementation - sign of a terminal branch of a node from a previous recursion cycle
    if (maxRemainingEntropy.toVector.map(_._2).count(x => x equals Double.NaN) > 0) {
      val terminalPath = path.map(x => (x._1.id, x._2))
      println("|----------> Partial terminal branch found, path: " + terminalPath.map(x => s"Attribute ${x._1} ---${x._2}--->").mkString(" "))
    } else if (optimalNode._2 == 0.0) {
      val terminalPath = path.map(x => (x._1.id, x._2)) :+ (optimalAttribute.id, 999)
      println("|----------> Terminal node found, path: " + terminalPath.map(x => s"Attribute ${x._1} ---${if (x._2 == 999) "<" else x._2.toString + "--->"}").mkString(" "))
    } else {
      println(s"\nPicking Attribute ${optimalAttribute.id} for current node.")
      val remainingAttributesAfter = remainingAttributes.filterNot(a => a.id == optimalAttribute.id)
      optimalAttribute.possibleValues.foreach(pv => findOptimalTree(remainingAttributesAfter, path :+ (optimalAttribute, pv)))
    }
  }
}
