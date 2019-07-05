import os
import math
import matplotlib.pyplot as plt

correctSH_path = "./data2/coefs/"
testSH_path = "./output/"



def takeSecond(elem):
    return elem[0]
 

def getNowSH(fp):
	fo = open(fp,"r")
	line = fo.readline()
	fo.close()
	line = line.split()
	lis = []
	for i in line:
		lis.append(float(i))
	return lis


class testSH:
	"""docstring for testSH"""
	name=''
	correct_sh = []
	test_sh = []
	loss_sh = []
	def printInfo(self):
		print(self.name)
		print(self.correct_sh)
		# for i in range(0,len(self.test_sh)):
		for i in range(0,100):
			print(self.test_sh[i])
		print(self.correct_sh)

	def getCorrectSH(self):
		correct_sh_fname = correctSH_path + self.name+".txt"
		fo = open(correct_sh_fname,"r")
		line = fo.readline()
		fo.close()
		line = line.split()
		lis = []
		for i in line:
			lis.append(float(i))
		return lis

	

	def getTestSH(self):
		print("getTestSH alert")
		
		filelis =[]
		for filename in os.listdir(testSH_path):
			# print(filename)
			target_name = self.name[:-2]+self.name[-1]
			fp = os.path.join(testSH_path, filename)
			if os.path.isfile(fp) and target_name in filename:
				filelis.append(fp)
		for f in filelis:
			lis = []
			los_lis =[]
			# print("--=-=",f)
			epoch = (int(f[:-4].split('_')[-1])+1)/200
			epoch = (int(epoch))
			now_sh = getNowSH(f)
			lis.append(epoch)
			lis.append(now_sh[0])
			lis.append(now_sh[1])
			lis.append(now_sh[2])
			# print(lis)
			x = now_sh[0]-self.correct_sh[0]
			y = now_sh[0]-self.correct_sh[0]
			z = now_sh[0]-self.correct_sh[0]
			m = round(abs(x)+abs(y)+abs(z),2)
			# print(m)
			lis.append(m)
			self.test_sh.append(lis)
		print(len(self.test_sh))

	def sortSH(self):
		self.test_sh.sort(key=takeSecond)

	def drawSH(self):
		x = []
		y1 = []
		y2 = []
		y3 = []
		y4 = []
		y5 = []
		y6 = []
		y7 = []
		for i in range(len(self.test_sh)):
			x.append(self.test_sh[i][0])
			y1.append(self.test_sh[i][1])
			y2.append(self.test_sh[i][2])
			y3.append(self.test_sh[i][3])
			y4.append(self.test_sh[i][4])
			y5.append(self.correct_sh[0])
			y6.append(self.correct_sh[1])
			y7.append(self.correct_sh[2])
		plt.figure(figsize=(8,4)) #创建绘图对象
		plt.plot(x,y1,"b-",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
		plt.plot(x,y5,"b--",linewidth=1) 
		plt.plot(x,y2,"g-",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
		plt.plot(x,y6,"g--",linewidth=1) 
		plt.plot(x,y3,"y-",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
		plt.plot(x,y7,"y--",linewidth=1) 
		plt.plot(x,y4,"r-",linewidth=1) 
		plt.xlabel("Time(s)") #X轴标签
		plt.ylabel("Volt")  #Y轴标签
		plt.title("Line plot") #图标题
		# plt.show()  #显示图
		plt.savefig("y2.jpg") #保存图



	def __init__(self, name):
		self.name = name  # name : pano_awhsjdhsd_3
		tmp_lis = self.getCorrectSH()
		for i in tmp_lis:
			self.correct_sh.append(i)
		self.getTestSH()
		self.sortSH()

	

		




def main():
	#pano_aqaivaniwrwnfs3_0_107999.txt
	# pano_aqrfwqryfzwuxk6
	fname = "pano_aqrfwqryfzwuxk6"
	fname = fname[:-1]+"_"+fname[-1]
	print (fname)
	sh1 = testSH(fname)
	# sh1.printInfo()
	sh1.drawSH()





if __name__ == '__main__':
    main()