# Explaination
## Simple Office task

+ **Task Objective**: The agent should place the book on the paper in its proper position on the shelf, then pick up the seal from the ink pad and stamp it in the bottom right corner of the paper. Besides, the agent should not move any books on the shelf originally.
+ **Environment**: An agent, a table where the event occurs, a shelf that already has three books, a seal, a cylindrical ink pad, a book that needs to be inserted into the shelf,a piece of paper that needs to be stamped,  and simulated digital components.
+ **Success Criteria**: 
  + The book on the paper was inserted into the correct place, which checked by both position and quaternion.
  + The bottom right corner of the paper was properly stamped.
  + The books on the shelf originally was not moved.
+ **Reward Structure**: 
  + Dense reward for success.
  + Dense reward for having book reach insert target.
  + Dense reward for having seal stamped on the goal marker.
  + Penality for moving any book on the shelf originally.
  + Progressive reward for having book close to the insert target by checking the book's position and quaternion.
  + Progressive reward for having seal close to goal marker.
+ **How to Randomize the Position of the Insert Target**:
  + The positions of the three books on the shelf originally are randomized in a range.
  + Three books make 4 gaps with the shelf. The insert target is the middle of the **largest** gap.
+ **The Penalty for moving the books on the shelf originally**: 
  + I'd like to train the agent to distinguish what can be move and what cannot.
  + However, I decide to implement it as reward to normalize the reward because of the given `compute_normalized_dense_reward` method.
+ **Fascinating Aspects that I Eager to Explore**:
  + The agent is digitally controlled to insert the book into the position specified by the digital signal.
  + Define a task that the stamp is returned to the ink pad after completing the stamping procedure.
  + Accurately identify the `stamp` action?
  + Make a system that can handle cases where the grasped objects are in motion during manipulation(e.g., in logistics sorting)?
