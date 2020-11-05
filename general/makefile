CC=gcc
CFLAGS=-I /home/rhiba/packages/include/
LDFLAGS= -L/home/rhiba/packages/lib -lopenblas -lpthread -lm
LPFLAGS= lp_dev/liblpsolve55.a -lm -ldl
DEPS=matrix.h nnet.h split.h entails.h

#all: network_test
all: entails
all: CFLAGS += -c -O3 -Wall -Werror -pedantic -fPIC
all: LDFLAGS += -O3

#debug: network_test
debug: entails
debug: CFLAGS += -DDEBUG -g
debug: LDFLAGS += -DDEBUG -g


demo2: demo2.o
	$(CC) -o $@ $^ $(LPFLAGS)

demo3: demo3.o
	$(CC) -o $@ $^ $(LPFLAGS)

#network_test: matrix.o nnet.o network_test.o split.o
entails: matrix.o nnet.o entails.o split.o
	$(CC) -shared -o libentails.so $^ $(LDFLAGS) $(LPFLAGS)

cblas_test:matrix.o cblas_test.o
	$(CC) -o $@ $^ $(LDFLAGS)

c.o: 
	$(CC) $(CFLAGS) $(LPFLAGS) $<  -o $@

#	rm -f *.o network_test
clean:
	rm -f *.o entails

