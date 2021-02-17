/*
 * Sorbonne Université 
 * Calcul de l'ensemble de mandelParabrot, Version parallèle
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>               /* compiler avec -lm */
#include <sys/time.h>
#include <arpa/inet.h>          /* htonl */
#include <mpi.h>
#include <string.h>

#include "rasterfile.h"

char info[] = "\
Usage:\n\
      ./mandelPara dimx dimy xmin ymin xmax ymax prof\n\
\n\
      dimx,dimy : dimensions de l'image à générer\n\
      xmin,ymin,xmax,ymax : domaine à calculer dans le plan complexe\n\
      prof : nombre maximal d'itérations\n\
\n\
Quelques exemples d'execution\n\
      ./mandelPara 3840 3840 0.35 0.355 0.353 0.358 200\n\
      ./mandelPara 3840 3840 -0.736 -0.184 -0.735 -0.183 500\n\
      ./mandelPara 3840 3840 -0.736 -0.184 -0.735 -0.183 300\n\
      ./mandelPara 3840 3840 -1.48478 0.00006 -1.48440 0.00044 100\n\
      ./mandelPara 3840 3840 -1.5 -0.1 -1.3 0.1 10000\n\
";

double wallclock_time()
{
	struct timeval tmp_time;
	gettimeofday(&tmp_time, NULL);
	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6);
}

unsigned char cos_composante(int i, double freq)
{
	double iD = i;
	iD = cos(iD / 255.0 * 2 * M_PI * freq);
	iD += 1;
	iD *= 128;
	return (unsigned char) iD;
}

/**
 *  Sauvegarde le tableau de données au format rasterfile
 */
void save_rasterfile(char *nom, int largeur, int hauteur, unsigned char *p)
{
	FILE *fd = fopen(nom, "w");
	if (fd == NULL) {
		perror("Error while opening output file. ");
		exit(1);
	}

	struct rasterfile file;
	file.ras_magic = htonl(RAS_MAGIC);
	file.ras_width = htonl(largeur);	            /* largeur en pixels de l'image */
	file.ras_height = htonl(hauteur);	        /* hauteur en pixels de l'image */
	file.ras_depth = htonl(8);	                /* profondeur de chaque pixel (1, 8 ou 24 ) */
	file.ras_length = htonl(largeur * hauteur);	/* taille de l'image en nb de bytes */
	file.ras_type = htonl(RT_STANDARD);	        /* type de fichier */
	file.ras_maptype = htonl(RMT_EQUAL_RGB);
	file.ras_maplength = htonl(256 * 3);
	fwrite(&file, sizeof(struct rasterfile), 1, fd);

	/* Palette de couleurs : composante rouge */
	for (int i = 255; i >= 0; i--) {
		unsigned char o = cos_composante(i, 13.0);
		fwrite(&o, sizeof(unsigned char), 1, fd);
	}

	/* Palette de couleurs : composante verte */
	for (int i = 255; i >= 0; i--) {
		unsigned char o = cos_composante(i, 5.0);
		fwrite(&o, sizeof(unsigned char), 1, fd);
	}

	/* Palette de couleurs : composante bleu */
	for (int i = 255; i >= 0; i--) {
		unsigned char o = cos_composante(i + 10, 7.0);
		fwrite(&o, sizeof(unsigned char), 1, fd);
	}

	fwrite(p, largeur * hauteur, sizeof(unsigned char), fd);
	fclose(fd);
}

/**
 * Étant donnée les coordonnées d'un point $c = a + ib$ dans le plan
 * complexe, la fonction retourne la couleur correspondante estimant
 * à quelle distance de l'ensemble de mandelParabrot le point est.
 * Soit la suite complexe définie par:
 * \begin{align}
 *     z_0     &= 0 \\
 *     z_{n+1} &= z_n^2 + c
 *   \end{align}
 * le nombre d'itérations que la suite met pour diverger est le
 * nombre $n$ pour lequel $|z_n| > 2$. 
 * Ce nombre est ramené à une valeur entre 0 et 255 correspond ainsi a 
 * une couleur dans la palette des couleurs.
 */
unsigned char xy2color(double a, double b, int prof)
{
	double x = 0;
	double y = 0;
	int i;
	for (i = 0; i < prof; i++) {
		/* garder la valeur précédente de x qui va etre ecrasé */
		double temp = x;
		/* nouvelles valeurs de x et y */
		double x2 = x * x;
		double y2 = y * y;
		x = x2 - y2 + a;
		y = 2 * temp * y + b;
		if (x2 + y2 > 4.0)
			break;
	}
	return (i == prof) ? 255 : (i % 255);
}

// Version BLOC par BLOC  (sans amerlioration de gestion des lignes)
int main(int argc, char *argv[]) {
	if (argc == 1)
		fprintf(stderr, "%s\n", info);

	double xmin = -2;     
	double ymin = -2;
	double xmax = 2;
	double ymax = 2;
	int w = 3840;       
	int h = 3840;
	int prof = 10000;    
	
	if (argc > 1)
		w = atoi(argv[1]);
	if (argc > 2)
		h = atoi(argv[2]);
	if (argc > 3)
		xmin = atof(argv[3]);
	if (argc > 4)
		ymin = atof(argv[4]);
	if (argc > 5)
		xmax = atof(argv[5]);
	if (argc > 6)
		ymax = atof(argv[6]);
	if (argc > 7)
		prof = atoi(argv[7]);

	double xinc = (xmax - xmin) / (w - 1);
	double yinc = (ymax - ymin) / (h - 1);

	int rank;
	int size;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int nb_lines = h/size;

	double debut = wallclock_time();
	if (rank==0){ 
		fprintf(stderr, "Domaine: [%g,%g] x [%g,%g]\n", xmin, ymin, xmax, ymax);
		fprintf(stderr, "Increment : %g, %g\n", xinc, yinc);
		fprintf(stderr, "Prof: %d\n", prof);
		fprintf(stderr, "Dim image: %d x %d\n", w, h);

		unsigned char *ima = malloc(w * h);
		if (ima == NULL) {
			perror("Erreur allocation mémoire du tableau : ");
			exit(1);
		}
		int k=0;
		// le processus 0 travail aussi avant de receptionner
		double y = ymin;
		for (int i=0; i < nb_lines; i++){
			double x = xmin;
			for (int j=0; j<w; j++){
				ima[j+k*w] =  xy2color(x, y, prof);
				x += xinc;
			}
			k+=1;
			y+=yinc;
		}
		for (int i=1; i < size; i++){
			// si c'est le dernier processeur et que N%P != 0 alors il doit faire un peu plus de travail que les autres
			if (i == size-1){
				nb_lines+=h%size;
			}
			for (int h1=0; h1<nb_lines; h1++){
				char msg[w*sizeof(unsigned char)];
				MPI_Recv(msg, w*sizeof(unsigned char), MPI_UNSIGNED_CHAR, i, 99 , MPI_COMM_WORLD, &status);
				for (int j=0; j<w; j++){
					ima[j+k*w] = msg[j];
				}
				k+=1;
			}	
		}
		
		double fin = wallclock_time();
		fprintf(stderr, "Temps total de calcul : %g sec\n", fin - debut);

		save_rasterfile("mandelPara.ras", w, h, ima);
	} else{
		double debut1 = wallclock_time();
		// nombre de ligne a traiter par processeur 
		double y = ymin + rank*nb_lines*yinc;
		// si c'est le dernier processeur et que N%P != 0 alors il doit faire un peu plus de travail que les autres
		if (rank == size-1){
			nb_lines+=h%size;
		}
		for (int i=0; i < nb_lines; i++){
			unsigned char *ima_int = malloc(w * sizeof(unsigned char));
			if (ima_int == NULL) {
				perror("Erreur allocation mémoire du tableau : ");
				exit(1);
			}
			double x = xmin;
			for (int j = 0; j < w; j++) {
				ima_int[j] = xy2color(x, y, prof);
				x += xinc;
			}
			y += yinc;
			MPI_Send(ima_int, w * sizeof(unsigned char), MPI_UNSIGNED_CHAR, 0, 99 , MPI_COMM_WORLD);
		}
		double fin1 = wallclock_time();
		fprintf(stderr, "Processeurs %d : Temps total de calcul : %g sec\n", rank, fin1 - debut1);
	}
	MPI_Finalize();
	return 0;
}