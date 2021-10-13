# Reinforcement learing :Jacks_rental_car_MDP -- Value Iteration

#### The description of the problem is in the section 4.3 of  _An Introduction of Reinforcement Learning_ Second edition by Sutton and Barto

Assumptions and Algorithm:
In Value Iteration, we first iterate state value V using (1) and then  determines the policies using greediy policy
![image](https://user-images.githubusercontent.com/88567713/137105157-891d6741-2b21-4af2-8840-c85fe3f7fb39.png)

AndV∗satisfies
![image](https://user-images.githubusercontent.com/88567713/137105301-0226b68b-37e1-4a80-a32d-0919fa97f6da.png)

We assume that state A and state B are independent.

###Full Algorithm (copied from the book mentioned above)

![image](https://user-images.githubusercontent.com/88567713/137105567-a3592600-2a71-4897-863d-7a645ab13e79.png)

The final result is the same as in the book on Page81
