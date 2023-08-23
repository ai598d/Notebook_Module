class compare:

  def IsClose(A,B,th):
    C = [A[0]-B[0],A[1]-B[1]]
    dist = np.linalg.norm(C)
    if(abs(dist)<(th) ):
      return True
    else:
      return False

  def CheckBad(Vmove,Omove,th):

    if(len(Vmove)>=len(Omove)):
      observation = len(Omove)
    else:
      observation = len(Vmove)


    pt = len(Vmove[0][0]) # number of points in a traj
    m = 0
    n = 0
    count = 0
    index = np.ones([observation])
    while(m<observation):


      while(n<pt):
        V_ppoints = Vmove[m,:,n]
        O_points  = Omove[m,:,n]

        if(compare.IsClose(V_ppoints,O_points,th) or compare.IsOutRange(V_ppoints) or compare.IsNegative(V_ppoints)):
          count = count+1
          n=pt # get outta loop
          index[m]=0

        else:
          count = count
          n=n+1
      n=0
      m=m+1

    return count,index

  def GetLabel(Vmove,Omove):
    count,index = compare.CheckBad(Vmove,Omove)
    return index

  def Bad_Counter(Vmove,Omove,thld):

    bad_count = np.zeros(len(thld))
    i=0
    while(i<len(bad_count)):
      bad_count[i] = compare.CheckBad(Vmove,Omove,thhld[i])[0]
      i=i+1

    return bad_count

  def IsOutRange(A):

    if(A[0]>1 or A[1]>1):
      return True
    else:
      return False

  def IsNegative(A):

    if(A[0]<0 or A[1]<0):
      return True
    else:
      return False




  def checkarray(move):

    '''
    input: move trajectory

    Return

      True: if theres a point in the move-trajectory greater than 1 or less than 0

      False: all the values are within 0-1 range

    '''
    length = len(move)

    i=0

    while(i<length):
      if( move[i]<0 or abs(move[i])>1 ):

        i = length
        return True

      else:
        i=i+1

    return False

  def CheckBadRange(array):
    '''
    input: a set of generated trajectories

    output: number and index of trajectories out of range (<0 or >1)
    '''

    Observation = array.shape[0]

    moveX = np.zeros(array.shape[2])
    moveY = np.zeros(array.shape[2])

    i=0

    count = 0
    index = []

    while(i<Observation):

      moveX = array[i,0,:]  # X coordinates of trajectory
      moveY = array[i,1,:]  # Y coordinates of trajectory

      if (compare.checkarray(moveX) or compare.checkarray(moveY)):
        count = count+1
        index.append(i)
        i=i+1

      else:

        i=i+1

    return count, index


  def StaticCheckBad(Vmove,Opos,th):

    observation = len(Vmove)
    pt = len(Vmove[0][0]) # number of points in a traj
    m = 0
    n = 0
    count = 0
    index = np.ones([observation])
    while(m<observation):


      while(n<pt):
        V_ppoints = Vmove[m,:,n]
        O_points  = Opos[m,:]

        if(compare.IsClose(V_ppoints,O_points,th) or compare.IsOutRange(V_ppoints) or compare.IsNegative(V_ppoints)):

          count = count+1
          n=pt # get outta loop
          index[m]=0

        else:

          count = count
          n=n+1

      n=0
      m=m+1

    return count,index
