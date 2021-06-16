## Dynamic load balancing using Master/Worker model. 
---

### Descriptions:
I implemented dynamic load balancing using the **Master-Worker** model. For example, there are independent tasks available to solve puzzles, but each job takes a variable amount of time. In that case, if tasks are distributed as fixed chunks, then there will be load imbalancing.  To overcome load imbalancing, Master rank distributes tasks as smaller chunk sizes on request from **Workers**. 

