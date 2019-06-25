

class A(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, output_dir = None):
        self.output_dir = output_dir



class B(A):
    def look(self):
        print('AAA')


    def look2(self):
        print(self.output_dir)



def C():
    print('c')



def main():
    bb = B()
    bb.look()
    bb2 = B(output_dir='123')
    bb2.look2()


if __name__ == '__main__':
    main()

