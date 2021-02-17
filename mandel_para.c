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

 // Version MAITRE ESCLAVE
int main(int argc, char *argv[]) {
	if (argc == 1)
		fprintf(stderr, "%s\n", info);

	// Valeurs par defaut de la fractale 
	double xmin = -2;     // Domaine de calcul dans le plan complexe 
	double ymin = -2;
	double xmax = 2;
	double ymax = 2;
	int w = 3840;         // Dimension de l'image (4K HTDV!) 
	int h = 3840;
	int prof = 10000;     // Profondeur d'iteration 
	
	
	// Recuperation des parametres 
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

	// Calcul des pas d'incrementation 
	double xinc = (xmax - xmin) / (w - 1);
	double yinc = (ymax - ymin) / (h - 1);

	int rank;
	int size;
	MPI_Status status;

	// PARRALELISATION 
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//int nb_lines = h/(size-1);
	int nb_lines = h/size;
	
	// debut du chronometrage
	double debut = wallclock_time();
	
	if (rank==0){ // Maitre, dans un premier temps le maitre ne travaille pas

		//affichage parametres pour verificatrion
		fprintf(stderr, "Domaine: [%g,%g] x [%g,%g]\n", xmin, ymin, xmax, ymax);
		fprintf(stderr, "Increment : %g, %g\n", xinc, yinc);
		fprintf(stderr, "Prof: %d\n", prof);
		fprintf(stderr, "Dim image: %d x %d\n", w, h);

		// Allocation memoire du tableau resultat
		unsigned char *ima = malloc(w * h);
		if (ima == NULL) {
			perror("Erreur allocation mémoire du tableau : ");
			exit(1);
		}

		int k = 0;	// indice d'envoie, la premiere ligne du bloc qu'on envoie a faire aux ecsclaves 
		int k_inter = 30; // nombre de ligne à faire par processeurs (taille d'un bloc) PARA

		// Tableau qui contient le point de départ et d'arrivé des blocs que les processeurs calculent
		int ** tab = malloc((size-1)*sizeof(int*));
		for(int i =0; i < size-1;i++){
			tab[i] = malloc(2*sizeof(int));
		} 

		// Le maitre envoie un premier travaille a chaque esclave, k lignes chacun
		for  (int i = 0; i < size-1; i++){
			int bornes[2] = {k,k+k_inter};
			MPI_Send(bornes, sizeof(int*), MPI_INT, i+1, 12 , MPI_COMM_WORLD);
			memcpy(tab[i], bornes, 2*sizeof(int)); // on stocke le bloc que chaque processeurs doit traiter 
			k+=k_inter;	// on incrémente la prochaine ligne à traiter (début du bloc suivant) 
		}

		// Le maitre va receptionner les buffers calculer des escalves, et si il reste du travail, leurs en donner d"autres
		int k1=0; // indice de recpetion, la premiere ligne du bloc calculé par les esclaves
		while (k1 < h){ // tant qu'on a pas receptionner toutes les lignes
			// réception des sous tableaux
			unsigned char *msg = malloc(w * k_inter);
			MPI_Recv(msg, w * k_inter* sizeof(unsigned char), MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, 99 , MPI_COMM_WORLD, &status);
			// on recupere les indices de debut et de fin que devait calcule le processeur du quel on recoit le message
			int dep = tab[status.MPI_SOURCE-1][0];
			int arr = tab[status.MPI_SOURCE-1][1];
			// on ajoute le sous tableau à notre tableau final
			int i1=0;
			for (int i=dep;i<arr;i++){
				for (int j=0; j<w; j++){
					ima[j+i*w] = msg[j+i1*w];
				}	
				i1++;
			}
			// si on a pas envoyé toutes les lignes, on donne du travail au processeurs qui vient de finir
			if (k!=h){	
				// si on a pas envoyé toutes les lignes et qu'il reste plus de k_inter ligne à faire
				if (k+k_inter  <= h){
					// on envoie les k_inter lignes suivantes a faire
					int bornes[2] = {k,k+k_inter};
					MPI_Ssend(bornes, sizeof(int*), MPI_INT, status.MPI_SOURCE, 12 , MPI_COMM_WORLD);
					memcpy(tab[status.MPI_SOURCE-1], bornes, 2*sizeof(int));
					k+=k_inter;
				} 
				// sinon on envoie les dernieres lignes a faire 
				else if(h-k != 0 && h-k > 0 && h-k < k_inter){ // si N%P != 0
					// on envoie le restant a faire
					int restant = h - k;
					int bornes[2] = {k,k+restant};
					MPI_Ssend(bornes, sizeof(int*), MPI_INT, status.MPI_SOURCE, 12 , MPI_COMM_WORLD);
					memcpy(tab[status.MPI_SOURCE-1], bornes, 2*sizeof(int));
					k+=restant;
				}
			}
			k1+=k_inter;
		}

		// tout le travail est fait, on previent les esclaves
		for (int i =0; i < size-1; i++){
			int bornes[2] = {k,-1};
			MPI_Ssend(bornes, 2*sizeof(int), MPI_INT, i+1, 12 , MPI_COMM_WORLD);
		}
		
		// fin du chronometrage 
		double fin = wallclock_time();
		fprintf(stderr, "Fin du maître : %g sec\n", fin - debut);
		// Sauvegarde de l'image dans le fichier resultat "mandelPara.ras"
		save_rasterfile("mandelPara.ras", w, h, ima);
	} else{
		// Esclaves
		double debut1 = wallclock_time();
		int k_dep = 0;
		int k_arr = 0;
		int k_inter  = 0;

		// tant que les esclaves n'ont pas le signal d'arret
		while(k_arr != -1){
			// ils attendents leurs travaillent a faire
			int bornes[2*sizeof(int)];
			MPI_Recv(bornes, 2*sizeof(int), MPI_INT, 0, 12 , MPI_COMM_WORLD, &status);
			k_dep = bornes[0];
			k_arr = bornes[1];
			// si ce n'est pas un signal d'arret
			if (k_arr != -1){
				//	ils calculent leurs bloc 
				k_inter  = k_arr - k_dep;
				unsigned char *ima = malloc(w * k_inter);
				double y = ymin + k_dep*yinc;
				for (int i=0; i < k_inter; i++){
					double x = xmin;
					for (int j = 0; j < w; j++) {
						ima[j+i*w] = xy2color(x, y, prof);
						x += xinc;
					}
					y += yinc;
				}
				// et envoie le resultat au maitre
				MPI_Ssend(ima, w * k_inter* sizeof(unsigned char), MPI_UNSIGNED_CHAR, 0, 99 , MPI_COMM_WORLD);
			}
		}
		double fin1 = wallclock_time();
		fprintf(stderr, "Processeurs %d : Temps total de calcul : %g sec\n", rank, fin1 - debut1);
	}

	MPI_Finalize();
	return 0;
}

/* 
 * Partie principale: en chaque point de la grille, appliquer xy2color
 */
/* // Version LIGNES par LIGNES
int main(int argc, char *argv[]) {
	if (argc == 1)
		fprintf(stderr, "%s\n", info);

	// Valeurs par defaut de la fractale 
	double xmin = -2;     // Domaine de calcul dans le plan complexe 
	double ymin = -2;
	double xmax = 2;
	double ymax = 2;
	int w = 3840;         // Dimension de l'image (4K HTDV!) 
	int h = 3840;
	int prof = 10000;     // Profondeur d'iteration 
	
	
	// Recuperation des parametres 
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

	// Calcul des pas d'incrementation 
	double xinc = (xmax - xmin) / (w - 1);
	double yinc = (ymax - ymin) / (h - 1);

	int rank;
	int size;
	MPI_Status status;

	// PARRALELISATION 
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//int nb_lines = h/(size-1);
	int nb_lines = h/size;
	
	// debut du chronometrage
	double debut = wallclock_time();
	if (rank==0){
		//affichage parametres pour verificatrion
		fprintf(stderr, "Domaine: [%g,%g] x [%g,%g]\n", xmin, ymin, xmax, ymax);
		fprintf(stderr, "Increment : %g, %g\n", xinc, yinc);
		fprintf(stderr, "Prof: %d\n", prof);
		fprintf(stderr, "Dim image: %d x %d\n", w, h);

		// Allocation memoire du tableau resultat
		unsigned char *ima = malloc(w * h);
		if (ima == NULL) {
			perror("Erreur allocation mémoire du tableau : ");
			exit(1);
		}
		int i=0;
		double y = ymin + (rank)*yinc;
		for (int h1=0; h1<h; h1++){
			unsigned char msg[w*sizeof(unsigned char)];
			if (h1%size == 0){
				double x = xmin;
				for (int j = 0; j < w; j++) {
					msg[j] = xy2color(x, y, prof);
					x += xinc;
				}
				y += (yinc*size);
			} else{
			MPI_Recv(msg, w*sizeof(unsigned char), MPI_UNSIGNED_CHAR, i+1, 99 , MPI_COMM_WORLD, &status);
			}
			
			for (int j=0; j<w; j++){
				ima[j+h1*w] = msg[j];
			}
			i++;
			if (i==size-1){
				i=0;
			}
		}
		// fin du chronometrage 
		double fin = wallclock_time();
		fprintf(stderr, "Temps total de calcul : %g sec\n", fin - debut);

		// Sauvegarde de l'image dans le fichier resultat "mandelPara.ras"
		save_rasterfile("mandelPara.ras", w, h, ima);
	} else{
		double debut1 = wallclock_time();

		//double y = ymin + (rank-1)*yinc;
		double y = ymin + (rank)*yinc;
		
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
			//y += (yinc*(size-1));
			y += (yinc*size);
			MPI_Ssend(ima_int, w * sizeof(unsigned char), MPI_UNSIGNED_CHAR, 0, 99 , MPI_COMM_WORLD);
		}
		double fin1 = wallclock_time();
		fprintf(stderr, "Processeurs %d : Temps total de calcul : %g sec\n", rank, fin1 - debut1);
	}
	MPI_Finalize();
	return 0;
}
*/

// Version BLOC par BLOC  (sans amerlioration de gestion des lignes)
/*int main(int argc, char *argv[]) {
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
	int nb_lines = h/(size-1);

	double debut = wallclock_time();
	if (rank==0){ // doit aussi calc
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
		for (int i=0; i < (size-1); i++){
			for (int h1=0; h1<nb_lines; h1++){
				char msg[w*sizeof(unsigned char)];
				MPI_Recv(msg, w*sizeof(unsigned char), MPI_UNSIGNED_CHAR, (i+1), 99 , MPI_COMM_WORLD, &status);
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
		double y = ymin + (rank-1)*nb_lines*yinc;
		//for (int i=(rank-1)*nb_lines; i < (rank-1)*nb_lines + (nb_lines-1);i++){
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
}*/